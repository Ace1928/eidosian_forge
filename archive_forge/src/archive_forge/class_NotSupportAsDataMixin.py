from llvmlite import ir
from llvmlite import binding as ll
from numba.core import datamodel
import unittest
class NotSupportAsDataMixin(object):
    """Ensure as_data() and from_data() raise NotImplementedError.
    """

    def test_as_data_not_supported(self):
        fnty = ir.FunctionType(ir.VoidType(), [])
        function = ir.Function(self.module, fnty, name='test_as_data')
        builder = ir.IRBuilder()
        builder.position_at_end(function.append_basic_block())
        undef_value = ir.Constant(self.datamodel.get_value_type(), None)
        with self.assertRaises(NotImplementedError):
            data = self.datamodel.as_data(builder, undef_value)
        with self.assertRaises(NotImplementedError):
            rev_data = self.datamodel.from_data(builder, undef_value)