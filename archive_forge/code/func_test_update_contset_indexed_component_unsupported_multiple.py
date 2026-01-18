import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.collections import ComponentMap
from pyomo.common.log import LoggingIntercept
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.dae.misc import (
def test_update_contset_indexed_component_unsupported_multiple(self):
    m = ConcreteModel()
    m.t = ContinuousSet(bounds=(0, 10))
    m.i = Set(initialize=[1, 2, 3])
    m.s = Set(m.i, m.t)
    generate_finite_elements(m.t, 5)
    expansion_map = ComponentMap()
    with self.assertRaises(TypeError):
        update_contset_indexed_component(m.s, expansion_map)