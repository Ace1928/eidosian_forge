import re
from ._exceptions import ProgrammingError
class CursorDictRowsMixIn:
    """This is a MixIn class that causes all rows to be returned as
    dictionaries. This is a non-standard feature."""
    _fetch_type = 1