import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
def test_one_to_one(self):
    m = self.model()
    index_set = m.b0.index_set()
    index = None
    pred_map = {0: UnindexedComponent_set}
    location_set_map = get_location_set_map(index, index_set)
    self.assertEqual(pred_map, location_set_map)
    index_set = m.b1.index_set()
    index = 1
    pred_map = {0: m.time}
    location_set_map = get_location_set_map(index, index_set)
    self.assertSameMap(pred_map, location_set_map)
    index_set = m.b2.index_set()
    index = (1, 2)
    pred_map = {0: m.time, 1: m.space}
    location_set_map = get_location_set_map(index, index_set)
    self.assertSameMap(pred_map, location_set_map)