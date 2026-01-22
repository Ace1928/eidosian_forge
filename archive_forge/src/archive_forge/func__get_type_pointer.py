from . import model
from .commontypes import COMMON_TYPES, resolve_common_type
from .error import FFIError, CDefError
import weakref, re, sys
def _get_type_pointer(self, type, quals, declname=None):
    if isinstance(type, model.RawFunctionType):
        return type.as_function_pointer()
    if isinstance(type, model.StructOrUnionOrEnum) and type.name.startswith('$') and type.name[1:].isdigit() and (type.forcename is None) and (declname is not None):
        return model.NamedPointerType(type, declname, quals)
    return model.PointerType(type, quals)