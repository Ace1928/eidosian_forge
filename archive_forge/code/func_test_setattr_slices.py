import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_setattr_slices(self):
    init_sum = sum(self.m.b[:, :].c[:, :].x[:].value)
    init_vals = list(self.m.b[1, :].c[:, 4].x[:].value)
    self.m.b[1, :].c[:, 4].x[:].value = 0
    new_sum = sum(self.m.b[:, :].c[:, :].x[:].value)
    new_vals = list(self.m.b[1, :].c[:, 4].x[:].value)
    self.assertEqual(len(init_vals), len(new_vals))
    self.assertNotEqual(init_vals, new_vals)
    self.assertEqual(sum(new_vals), 0)
    self.assertEqual(init_sum - sum(init_vals), new_sum)
    _slice = self.m.b[...].c[...].x[:]
    with self.assertRaisesRegex(AttributeError, ".*VarData' object has no attribute 'bogus'"):
        _slice.bogus = 0
    _slice.attribute_errors_generate_exceptions = False
    _slice.bogus = 0