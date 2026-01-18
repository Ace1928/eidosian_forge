import pickle
import pyomo.common.unittest as unittest
from pyomo.environ import Var, Block, ConcreteModel, RangeSet, Set, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.set import normalize_index
def test_pickle_slices(self):
    m = self.m
    _slicer = m.b[1, :].c[:, 4].x
    _new_slicer = pickle.loads(pickle.dumps(_slicer))
    self.assertIsNot(_slicer, _new_slicer)
    self.assertIsNot(_slicer._call_stack, _new_slicer._call_stack)
    self.assertIs(type(_slicer._call_stack), type(_new_slicer._call_stack))
    self.assertEqual(len(_slicer._call_stack), len(_new_slicer._call_stack))
    ref = ['b[1,4].c[1,4].x', 'b[1,4].c[2,4].x', 'b[1,4].c[3,4].x', 'b[1,5].c[1,4].x', 'b[1,5].c[2,4].x', 'b[1,5].c[3,4].x', 'b[1,6].c[1,4].x', 'b[1,6].c[2,4].x', 'b[1,6].c[3,4].x']
    self.assertEqual([str(x) for x in _slicer], ref)
    self.assertEqual([str(x) for x in _new_slicer], ref)
    for x, y in zip(iter(_slicer), iter(_new_slicer)):
        self.assertIs(type(x), type(y))
        self.assertEqual(x.name, y.name)
        self.assertIsNot(x, y)