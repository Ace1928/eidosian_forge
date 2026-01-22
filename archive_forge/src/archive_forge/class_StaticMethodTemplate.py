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
class StaticMethodTemplate(templates.AbstractTemplate):
    key = (self.key, attr)

    def generic(self, args, kws):
        sig = disp_type.get_call_type(self.context, args, kws)
        return sig.replace(recvr=instance)