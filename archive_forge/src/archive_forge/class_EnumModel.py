from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.EnumMember)
@register_default(types.IntEnumMember)
class EnumModel(ProxyModel):
    """
    Enum members are represented exactly like their values.
    """

    def __init__(self, dmm, fe_type):
        super(EnumModel, self).__init__(dmm, fe_type)
        self._proxied_model = dmm.lookup(fe_type.dtype)