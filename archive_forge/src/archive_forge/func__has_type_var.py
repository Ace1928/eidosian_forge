import sys
import types
import typing
import typing_extensions
from mypy_extensions import _TypedDictMeta as _TypedDictMeta_Mypy
def _has_type_var(t):
    if t is None:
        return False
    elif is_union_type(t):
        return _union_has_type_var(t)
    elif is_tuple_type(t):
        return _tuple_has_type_var(t)
    elif is_generic_type(t):
        return _generic_has_type_var(t)
    elif is_callable_type(t):
        return _callable_has_type_var(t)
    else:
        return False