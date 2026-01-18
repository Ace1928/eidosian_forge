import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_update_contset_indexed_component_expressions_single(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.p = Param(m.t, default=3)
    m.v = Var(m.t, initialize=5)

    def _con1(m, i):
        return m.p[i] * m.v[i]
    m.con1 = Expression(m.t, rule=_con1)

    def _con2(m):
        return sum((m.v[i] for i in m.t))
    m.con2 = Expression(rule=_con2)
    expansion_map = ComponentMap()
    generate_finite_elements(m.t, 5)
    update_contset_indexed_component(m.v, expansion_map)
    update_contset_indexed_component(m.p, expansion_map)
    update_contset_indexed_component(m.con1, expansion_map)
    update_contset_indexed_component(m.con2, expansion_map)
    self.assertTrue(len(m.con1) == 6)
    self.assertEqual(m.con1[2](), 15)
    self.assertEqual(m.con1[8](), 15)
    self.assertEqual(m.con2(), 10)