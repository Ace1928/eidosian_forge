from gitdb.util import (
from gitdb.utils.encoding import force_text
from gitdb.exc import (
from itertools import chain
from functools import reduce
def _db_query(self, sha):
    """:return: database containing the given 20 byte sha
        :raise BadObject:"""
    try:
        return self._db_cache[sha]
    except KeyError:
        pass
    for db in self._dbs:
        if db.has_object(sha):
            self._db_cache[sha] = db
            return db
    raise BadObject(sha)