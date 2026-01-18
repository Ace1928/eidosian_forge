import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.scc_solver import (
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_with_zero_coefficients(self):
    """Test where the blocks we identify are incorrect if we don't filter
        out variables with coefficients of zero
        """
    m = pyo.ConcreteModel()
    m.x = pyo.Var([1, 2, 3], initialize=1.0)
    m.eq1 = pyo.Constraint(expr=m.x[1] + 2 * m.x[2] + 0 * m.x[3] == 7)
    m.eq2 = pyo.Constraint(expr=m.x[1] + pyo.log(m.x[1]) == 0)
    results = solve_strongly_connected_components(m)
    self.assertAlmostEqual(m.x[1].value, 0.56714329)
    self.assertAlmostEqual(m.x[2].value, 3.21642835)
    self.assertEqual(m.x[3].value, 1.0)