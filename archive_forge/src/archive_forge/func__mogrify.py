import re
from ._exceptions import ProgrammingError
def _mogrify(self, query, args=None):
    """Return query after binding args."""
    db = self._get_db()
    if isinstance(query, str):
        query = query.encode(db.encoding)
    if args is not None:
        if isinstance(args, dict):
            nargs = {}
            for key, item in args.items():
                if isinstance(key, str):
                    key = key.encode(db.encoding)
                nargs[key] = db.literal(item)
            args = nargs
        else:
            args = tuple(map(db.literal, args))
        try:
            query = query % args
        except TypeError as m:
            raise ProgrammingError(str(m))
    return query