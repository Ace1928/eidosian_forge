from .._compat import basestring
from .._compat import urlencode as _urlencode
def _is_two_tuple(item):
    return isinstance(item, (list, tuple)) and len(item) == 2