from llvmlite import ir, binding as ll
from numba.core import types, datamodel
from numba.core.datamodel.testing import test_factory
from numba.core.datamodel.manager import DataModelManager
from numba.core.datamodel.models import OpaqueModel
import unittest
def _test_as_arguments(self, fe_args):
    """
        Test round-tripping types *fe_args* through the default data model's
        argument conversion and unpacking logic.
        """
    dmm = datamodel.default_manager
    fi = datamodel.ArgPacker(dmm, fe_args)
    module = ir.Module()
    fnty = ir.FunctionType(ir.VoidType(), [])
    function = ir.Function(module, fnty, name='test_arguments')
    builder = ir.IRBuilder()
    builder.position_at_end(function.append_basic_block())
    args = [ir.Constant(dmm.lookup(t).get_value_type(), None) for t in fe_args]
    values = fi.as_arguments(builder, args)
    asargs = fi.from_arguments(builder, values)
    self.assertEqual(len(asargs), len(fe_args))
    valtys = tuple([v.type for v in values])
    self.assertEqual(valtys, fi.argument_types)
    expect_types = [a.type for a in args]
    got_types = [a.type for a in asargs]
    self.assertEqual(expect_types, got_types)
    fi.assign_names(values, ['arg%i' for i in range(len(fe_args))])
    builder.ret_void()
    ll.parse_assembly(str(module))