import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_update_contset_indexed_component_vars_multiple(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.t2 = ContinuousSet(initialize=[1, 2, 3])
    m.s = Set(initialize=[1, 2, 3])
    m.s2 = Set(initialize=[(1, 1), (2, 2)])
    m.v1 = Var(m.s, m.t, initialize=3)
    m.v2 = Var(m.s, m.t, m.t2, bounds=(4, 10), initialize={(1, 0, 1): 22, (2, 10, 2): 22})

    def _init(m, i, j, k):
        return i
    m.v3 = Var(m.t, m.s2, bounds=(-5, 5), initialize=_init)
    m.v4 = Var(m.s, m.t2, initialize=7, dense=True)
    m.v5 = Var(m.s2)
    expansion_map = ComponentMap()
    generate_finite_elements(m.t, 5)
    update_contset_indexed_component(m.v1, expansion_map)
    update_contset_indexed_component(m.v2, expansion_map)
    update_contset_indexed_component(m.v3, expansion_map)
    update_contset_indexed_component(m.v4, expansion_map)
    update_contset_indexed_component(m.v5, expansion_map)
    self.assertTrue(len(m.v1) == 18)
    self.assertTrue(len(m.v2) == 54)
    self.assertTrue(len(m.v3) == 12)
    self.assertTrue(len(m.v4) == 9)
    self.assertTrue(value(m.v1[1, 4]) == 3)
    self.assertTrue(m.v1[2, 2].ub is None)
    self.assertTrue(m.v1[3, 8].lb is None)
    self.assertTrue(value(m.v2[1, 0, 1]) == 22)
    self.assertTrue(m.v2[1, 2, 1].value is None)
    self.assertTrue(m.v2[2, 4, 3].lb == 4)
    self.assertTrue(m.v2[3, 8, 1].ub == 10)
    self.assertTrue(value(m.v3[2, 2, 2]) == 2)
    self.assertTrue(m.v3[4, 1, 1].lb == -5)
    self.assertTrue(m.v3[8, 2, 2].ub == 5)
    self.assertTrue(value(m.v3[6, 1, 1]) == 6)