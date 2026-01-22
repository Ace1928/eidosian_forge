from llvmlite import ir
from llvmlite import binding as ll
from numba.core import datamodel
import unittest
class SupportAsDataMixin(object):
    """Test as_data() and from_data()
    """

    def test_as_data(self):
        fnty = ir.FunctionType(ir.VoidType(), [])
        function = ir.Function(self.module, fnty, name='test_as_data')
        builder = ir.IRBuilder()
        builder.position_at_end(function.append_basic_block())
        undef_value = ir.Constant(self.datamodel.get_value_type(), None)
        data = self.datamodel.as_data(builder, undef_value)
        self.assertIsNot(data, NotImplemented, 'as_data returned NotImplemented')
        self.assertEqual(data.type, self.datamodel.get_data_type())
        rev_value = self.datamodel.from_data(builder, data)
        self.assertEqual(rev_value.type, self.datamodel.get_value_type())
        builder.ret_void()
        materialized = ll.parse_assembly(str(self.module))
        str(materialized)