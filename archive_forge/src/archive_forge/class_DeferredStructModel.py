from functools import partial
from collections import deque
from llvmlite import ir
from numba.core.datamodel.registry import register_default
from numba.core import types, cgutils
from numba.np import numpy_support
@register_default(types.DeferredType)
class DeferredStructModel(CompositeModel):

    def __init__(self, dmm, fe_type):
        super(DeferredStructModel, self).__init__(dmm, fe_type)
        self.typename = 'deferred.{0}'.format(id(fe_type))
        self.actual_fe_type = fe_type.get()

    def get_value_type(self):
        return ir.global_context.get_identified_type(self.typename + '.value')

    def get_data_type(self):
        return ir.global_context.get_identified_type(self.typename + '.data')

    def get_argument_type(self):
        return self._actual_model.get_argument_type()

    def as_argument(self, builder, value):
        inner = self.get(builder, value)
        return self._actual_model.as_argument(builder, inner)

    def from_argument(self, builder, value):
        res = self._actual_model.from_argument(builder, value)
        return self.set(builder, self.make_uninitialized(), res)

    def from_data(self, builder, value):
        self._define()
        elem = self.get(builder, value)
        value = self._actual_model.from_data(builder, elem)
        out = self.make_uninitialized()
        return self.set(builder, out, value)

    def as_data(self, builder, value):
        self._define()
        elem = self.get(builder, value)
        value = self._actual_model.as_data(builder, elem)
        out = self.make_uninitialized(kind='data')
        return self.set(builder, out, value)

    def from_return(self, builder, value):
        return value

    def as_return(self, builder, value):
        return value

    def get(self, builder, value):
        return builder.extract_value(value, [0])

    def set(self, builder, value, content):
        return builder.insert_value(value, content, [0])

    def make_uninitialized(self, kind='value'):
        self._define()
        if kind == 'value':
            ty = self.get_value_type()
        else:
            ty = self.get_data_type()
        return ir.Constant(ty, ir.Undefined)

    def _define(self):
        valty = self.get_value_type()
        self._define_value_type(valty)
        datty = self.get_data_type()
        self._define_data_type(datty)

    def _define_value_type(self, value_type):
        if value_type.is_opaque:
            value_type.set_body(self._actual_model.get_value_type())

    def _define_data_type(self, data_type):
        if data_type.is_opaque:
            data_type.set_body(self._actual_model.get_data_type())

    @property
    def _actual_model(self):
        return self._dmm.lookup(self.actual_fe_type)

    def traverse(self, builder):
        return [(self.actual_fe_type, lambda value: builder.extract_value(value, [0]))]