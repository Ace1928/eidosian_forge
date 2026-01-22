from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
class PythonFunctionDescriptor(FunctionDescriptor):
    """
    A FunctionDescriptor subclass for Numba-compiled functions.
    """
    __slots__ = ()

    @classmethod
    def from_specialized_function(cls, func_ir, typemap, restype, calltypes, mangler, inline, noalias, abi_tags):
        """
        Build a FunctionDescriptor for a given specialization of a Python
        function (in nopython mode).
        """
        return cls._from_python_function(func_ir, typemap, restype, calltypes, native=True, mangler=mangler, inline=inline, noalias=noalias, abi_tags=abi_tags)

    @classmethod
    def from_object_mode_function(cls, func_ir):
        """
        Build a FunctionDescriptor for an object mode variant of a Python
        function.
        """
        typemap = defaultdict(lambda: types.pyobject)
        calltypes = typemap.copy()
        restype = types.pyobject
        return cls._from_python_function(func_ir, typemap, restype, calltypes, native=False)