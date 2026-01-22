from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.Complex)
class ComplexModel(StructModel):
    _element_type = NotImplemented

    def __init__(self, dmm, fe_type):
        members = [('real', fe_type.underlying_float), ('imag', fe_type.underlying_float)]
        super(ComplexModel, self).__init__(dmm, fe_type, members)