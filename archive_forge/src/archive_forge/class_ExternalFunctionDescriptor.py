from collections import defaultdict
import importlib
from numba.core import types, itanium_mangler
from numba.core.utils import _dynamic_modname, _dynamic_module
class ExternalFunctionDescriptor(FunctionDescriptor):
    """
    A FunctionDescriptor subclass for opaque external functions
    (e.g. raw C functions).
    """
    __slots__ = ()

    def __init__(self, name, restype, argtypes):
        args = ['arg%d' % i for i in range(len(argtypes))]

        def mangler(a, x, abi_tags, uid=None):
            return a
        super(ExternalFunctionDescriptor, self).__init__(native=True, modname=None, qualname=name, unique_name=name, doc='', typemap=None, restype=restype, calltypes=None, args=args, kws=None, mangler=mangler, argtypes=argtypes)