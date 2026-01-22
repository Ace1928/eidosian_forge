from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.Optional)
class OptionalModel(StructModel):

    def __init__(self, dmm, fe_type):
        members = [('data', fe_type.type), ('valid', types.boolean)]
        self._value_model = dmm.lookup(fe_type.type)
        super(OptionalModel, self).__init__(dmm, fe_type, members)

    def get_return_type(self):
        return self._value_model.get_return_type()

    def as_return(self, builder, value):
        raise NotImplementedError

    def from_return(self, builder, value):
        return self._value_model.from_return(builder, value)

    def traverse(self, builder):

        def get_data(value):
            valid = get_valid(value)
            data = self.get(builder, value, 'data')
            return builder.select(valid, data, ir.Constant(data.type, None))

        def get_valid(value):
            return self.get(builder, value, 'valid')
        return [(self.get_type('data'), get_data), (self.get_type('valid'), get_valid)]