import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_update_contset_indexed_component_piecewise_single(self):
    x = [0.0, 1.5, 3.0, 5.0]
    y = [1.1, -1.1, 2.0, 1.1]
    model = ConcreteModel()
    model.t = ContinuousSet(bounds=(0, 10))
    model.x = Var(model.t, bounds=(min(x), max(x)))
    model.y = Var(model.t)
    model.fx = Piecewise(model.t, model.y, model.x, pw_pts=x, pw_constr_type='EQ', f_rule=y)
    self.assertEqual(len(model.fx), 2)
    expansion_map = ComponentMap()
    generate_finite_elements(model.t, 5)
    update_contset_indexed_component(model.fx, expansion_map)
    self.assertEqual(len(model.fx), 6)
    self.assertEqual(len(model.fx[2].SOS2_constraint), 3)