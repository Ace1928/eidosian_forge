import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_nonterminal_slice(self):
    _slicer = self.m.b[:, 4].x
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['b[1,4].x', 'b[2,4].x', 'b[3,4].x'])
    _slicer = self.m.b[:, 4].x[7]
    self.assertIsInstance(_slicer, IndexedComponent_slice)
    ans = [str(x) for x in _slicer]
    self.assertEqual(ans, ['b[1,4].x[7]', 'b[2,4].x[7]', 'b[3,4].x[7]'])