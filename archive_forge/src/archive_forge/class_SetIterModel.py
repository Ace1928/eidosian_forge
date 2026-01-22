from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.SetIter)
class SetIterModel(StructModel):

    def __init__(self, dmm, fe_type):
        payload_type = types.SetPayload(fe_type.container)
        members = [('meminfo', types.MemInfoPointer(payload_type)), ('index', types.EphemeralPointer(types.intp))]
        super(SetIterModel, self).__init__(dmm, fe_type, members)