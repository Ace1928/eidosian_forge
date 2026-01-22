from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.RangeType)
class RangeModel(StructModel):

    def __init__(self, dmm, fe_type):
        int_type = fe_type.iterator_type.yield_type
        members = [('start', int_type), ('stop', int_type), ('step', int_type)]
        super(RangeModel, self).__init__(dmm, fe_type, members)