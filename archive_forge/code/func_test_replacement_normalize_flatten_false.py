import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
def test_replacement_normalize_flatten_false(self):
    m = self.model()
    index = (1, 0, ('a', 1))
    location_set_map = {0: m.time, 1: m.space, 2: m.d_2}
    sets = ComponentSet((m.d_2,))
    pred_index = (1, 0, slice(None))
    normalize_index.flatten = False
    new_index = replace_indices(index, location_set_map, sets)
    self.assertEqual(new_index, pred_index)
    normalize_index.flatten = True