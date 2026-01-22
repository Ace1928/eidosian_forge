import os.path
import pickle
import sys
from rdkit import RDConfig
from rdkit.VLib.Supply import SupplyNode
class _lazyDataSeq:
    """
    These classes are used to speed up (a lot) the process of
    pulling pickled objects from PostgreSQL databases.  Instead of
    having to use all of PgSQL's typechecking, we'll make a lot of
    assumptions about what's coming out of the Db and its layout.
    The results can lead to drastic improvements in performance.
    
    """

    def __init__(self, cursor, cmd, pickleCol=1, depickle=1, klass=None):
        self.cursor = cursor
        self.cmd = cmd
        self._first = 0
        self._pickleCol = pickleCol
        self._depickle = depickle
        self._klass = klass

    def _validate(self):
        curs = self.cursor
        if not curs or curs.closed or curs.conn is None or (curs.res.resultType != sql.RESULT_DQL and curs.closed is None):
            raise ValueError('bad cursor')
        if curs.res.nfields and curs.res.nfields < 2:
            raise ValueError('invalid number of results returned (%d), must be at least 2' % curs.res.nfields)
        desc1 = curs.description[self._pickleCol]
        ftv = desc1[self._pickleCol].value
        if ftv != sql.BINARY:
            raise TypeError('pickle column (%d) of bad type' % self._pickleCol)

    def __iter__(self):
        try:
            self.cursor.execute(self.cmd)
        except Exception:
            import traceback
            traceback.print_exc()
            print('COMMAND:', self.cmd)
            raise
        self._first = 1
        self._validate()
        return self

    def next(self):
        curs = self.cursor
        if not curs or curs.closed or curs.conn is None or (curs.res is None) or (curs.res.resultType != sql.RESULT_DQL and curs.closed is None):
            raise StopIteration
        if not self._first:
            res = curs.conn.conn.query('fetch 1 from "%s"' % self.cursor.name)
            if res.ntuples == 0:
                raise StopIteration
            else:
                if res.nfields < 2:
                    raise ValueError('bad result: %s' % str(res))
                t = [res.getvalue(0, x) for x in range(res.nfields)]
                val = t[self._pickleCol]
        else:
            t = curs.fetchone()
            val = str(t[self._pickleCol])
            self._first = 0
        if self._depickle:
            if not self._klass:
                fp = pickle.loads(val)
            else:
                fp = self._klass(val)
            fields = list(t)
            del fields[self._pickleCol]
            fp._fieldsFromDb = fields
        else:
            fp = list(t)
        return fp