import re
from ._exceptions import ProgrammingError
class CursorUseResultMixIn:
    """This is a MixIn class which causes the result set to be stored
    in the server and sent row-by-row to client side, i.e. it uses
    mysql_use_result(). You MUST retrieve the entire result set and
    close() the cursor before additional queries can be performed on
    the connection."""

    def _get_result(self):
        return self._get_db().use_result()

    def fetchone(self):
        """Fetches a single row from the cursor."""
        self._check_executed()
        r = self._fetch_row(1)
        if not r:
            return None
        self.rownumber = self.rownumber + 1
        return r[0]

    def fetchmany(self, size=None):
        """Fetch up to size rows from the cursor. Result set may be smaller
        than size. If size is not defined, cursor.arraysize is used."""
        self._check_executed()
        r = self._fetch_row(size or self.arraysize)
        self.rownumber = self.rownumber + len(r)
        return r

    def fetchall(self):
        """Fetches all available rows from the cursor."""
        self._check_executed()
        r = self._fetch_row(0)
        self.rownumber = self.rownumber + len(r)
        return r

    def __iter__(self):
        return self

    def next(self):
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row
    __next__ = next