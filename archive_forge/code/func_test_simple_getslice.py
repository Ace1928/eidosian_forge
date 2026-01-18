import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_simple_getslice(self):
    _slicer = self.m.b[:, 4]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['b[1,4]', 'b[2,4]', 'b[3,4]'])
    _slicer = self.m.b[1, 4].c[:, 4]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['b[1,4].c[1,4]', 'b[1,4].c[2,4]', 'b[1,4].c[3,4]'])