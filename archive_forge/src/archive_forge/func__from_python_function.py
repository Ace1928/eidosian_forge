from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
@classmethod
def _from_python_function(cls, func_ir, typemap, restype, calltypes, native, mangler=None, inline=False, noalias=False, abi_tags=()):
    qualname, unique_name, modname, doc, args, kws, global_dict = cls._get_function_info(func_ir)
    self = cls(native, modname, qualname, unique_name, doc, typemap, restype, calltypes, args, kws, mangler=mangler, inline=inline, noalias=noalias, global_dict=global_dict, abi_tags=abi_tags, uid=func_ir.func_id.unique_id)
    return self