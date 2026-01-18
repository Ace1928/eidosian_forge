from io import StringIO
import logging
from os.path import join, normpath
import pickle
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.core.expr.compare import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction, GDP_Error
from pyomo.gdp.tests.common_tests import (
from pyomo.gdp.tests.models import make_indexed_equality_model
from pyomo.repn import generate_standard_repn
def test_only_multiple_bigm_bound_constraints_arg_Ms(self):
    m = self.make_model()
    mbm = TransformationFactory('gdp.mbigm')
    Ms = {m.d1: 1050, m.d2.func: (-2000, 1200), None: 4000}
    mbm.apply_to(m, only_mbigm_bound_constraints=True, bigM=Ms)
    cons = mbm.get_transformed_constraints(m.d1.x1_bounds)
    self.assertEqual(len(cons), 2)
    self.check_pretty_bound_constraints(cons[0], m.x1, {m.d1: 0.5, m.d2: 0.65, m.d3: 2}, lb=True)
    self.check_pretty_bound_constraints(cons[1], m.x1, {m.d1: 2, m.d2: 3, m.d3: 10}, lb=False)
    cons = mbm.get_transformed_constraints(m.d1.x2_bounds)
    self.assertEqual(len(cons), 2)
    self.check_pretty_bound_constraints(cons[0], m.x2, {m.d1: 0.75, m.d2: 3, m.d3: 0.55}, lb=True)
    self.check_pretty_bound_constraints(cons[1], m.x2, {m.d1: 3, m.d2: 10, m.d3: 1}, lb=False)
    self.check_traditionally_bigmed_constraints(m, mbm, {m.d1: (-1050, 1050), m.d2: (-2000, 1200), m.d3: (-4000, 4000)})