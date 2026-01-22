import types
from jedi import debug
def _safe_hasattr(obj, name):
    return _check_class(type(obj), name) is not _sentinel