from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.Boolean)
@register_default(types.BooleanLiteral)
class BooleanModel(DataModel):
    _bit_type = ir.IntType(1)
    _byte_type = ir.IntType(8)

    def get_value_type(self):
        return self._bit_type

    def get_data_type(self):
        return self._byte_type

    def get_return_type(self):
        return self.get_data_type()

    def get_argument_type(self):
        return self.get_data_type()

    def as_data(self, builder, value):
        return builder.zext(value, self.get_data_type())

    def as_argument(self, builder, value):
        return self.as_data(builder, value)

    def as_return(self, builder, value):
        return self.as_data(builder, value)

    def from_data(self, builder, value):
        ty = self.get_value_type()
        resalloca = cgutils.alloca_once(builder, ty)
        cond = builder.icmp_unsigned('==', value, value.type(0))
        with builder.if_else(cond) as (then, otherwise):
            with then:
                builder.store(ty(0), resalloca)
            with otherwise:
                builder.store(ty(1), resalloca)
        return builder.load(resalloca)

    def from_argument(self, builder, value):
        return self.from_data(builder, value)

    def from_return(self, builder, value):
        return self.from_data(builder, value)