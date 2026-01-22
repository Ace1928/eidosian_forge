from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.MemInfoPointer)
class MemInfoModel(OpaqueModel):

    def inner_models(self):
        return [self._dmm.lookup(self._fe_type.dtype)]

    def has_nrt_meminfo(self):
        return True

    def get_nrt_meminfo(self, builder, value):
        return value