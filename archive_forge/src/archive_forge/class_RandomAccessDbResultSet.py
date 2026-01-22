import sys
from rdkit.Dbase import DbInfo
class RandomAccessDbResultSet(DbResultBase):
    """ Supports random access """

    def __init__(self, *args, **kwargs):
        DbResultBase.__init__(self, *args, **kwargs)
        self.results = []
        self.seen = []
        self._pos = -1

    def Reset(self):
        self._pos = -1
        if self.cursor is not None:
            DbResultBase.Reset(self)

    def _finish(self):
        if self.cursor:
            r = self.cursor.fetchone()
            while r:
                if self.transform is not None:
                    r = self.transform(r)
                if self.removeDups >= 0:
                    v = r[self.removeDups]
                    if v not in self.seen:
                        self.seen.append(v)
                        self.results.append(r)
                else:
                    self.results.append(r)
                r = self.cursor.fetchone()
            self.cursor = None

    def __getitem__(self, idx):
        if idx < 0:
            raise IndexError('negative indices not supported')
        if self.cursor is None:
            if len(self.results):
                if idx >= len(self.results):
                    raise IndexError('index %d too large (%d max)' % (idx, len(self.results)))
            else:
                raise ValueError('Invalid cursor')
        while idx >= len(self.results):
            r = None
            while r is None:
                r = self.cursor.fetchone()
                if not r:
                    self.cursor = None
                    raise IndexError('index %d too large (%d max)' % (idx, len(self.results)))
                if self.transform is not None:
                    r = self.transform(r)
                if self.removeDups >= 0:
                    v = r[self.removeDups]
                    if v in self.seen:
                        r = None
                    else:
                        self.results.append(r)
                        self.seen.append(v)
                else:
                    self.results.append(r)
        return self.results[idx]

    def __len__(self):
        if self.results is None:
            raise ValueError('len() not supported for noMemory Results Sets')
        self._finish()
        return len(self.results)

    def next(self):
        self._pos += 1
        res = None
        if self._pos < len(self):
            res = self.results[self._pos]
        else:
            raise StopIteration
        return res
    __next__ = next