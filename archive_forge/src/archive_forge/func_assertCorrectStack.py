import pyomo.common.unittest as unittest
from pyomo.core.base.indexed_component import normalize_index
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice
from pyomo.core.base.global_set import UnindexedComponent_set
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.util.slices import (
def assertCorrectStack(self, comp, pred_stack, context=None):
    act_stack = get_component_call_stack(comp, context=None)
    self.assertSameStack(pred_stack, act_stack)