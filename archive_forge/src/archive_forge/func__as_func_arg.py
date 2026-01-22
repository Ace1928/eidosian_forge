from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _as_func_arg(self, type, quals):
    if isinstance(type, model.ArrayType):
        return model.PointerType(type.item, quals)
    elif isinstance(type, model.RawFunctionType):
        return type.as_function_pointer()
    else:
        return type