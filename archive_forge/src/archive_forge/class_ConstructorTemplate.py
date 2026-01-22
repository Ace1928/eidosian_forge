import inspect
import operator
import types as pytypes
import typing as pt
from collections import OrderedDict
from collections.abc import Sequence
from llvmlite import ir as llvmir
from numba import njit
from numba.core import cgutils, errors, imputils, types, utils
from numba.core.datamodel import default_manager, models
from numba.core.registry import cpu_target
from numba.core.typing import templates
from numba.core.typing.asnumbatype import as_numba_type
from numba.core.serialize import disable_pickling
from numba.experimental.jitclass import _box
class ConstructorTemplate(templates.AbstractTemplate):
    """
    Base class for jitclass constructor templates.
    """

    def generic(self, args, kws):
        instance_type = self.key.instance_type
        ctor = instance_type.jit_methods['__init__']
        boundargs = (instance_type.get_reference_type(),) + args
        disp_type = types.Dispatcher(ctor)
        sig = disp_type.get_call_type(self.context, boundargs, kws)
        if not isinstance(sig.return_type, types.NoneType):
            raise errors.NumbaTypeError(f"__init__() should return None, not '{sig.return_type}'")
        out = templates.signature(instance_type, *sig.args[1:])
        return out