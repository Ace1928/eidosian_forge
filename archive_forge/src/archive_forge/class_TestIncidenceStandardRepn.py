import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
class TestIncidenceStandardRepn(unittest.TestCase, _TestIncidence, _TestIncidenceLinearOnly, _TestIncidenceLinearCancellation):

    def _get_incident_variables(self, expr, **kwds):
        method = IncidenceMethod.standard_repn
        return get_incident_variables(expr, method=method, **kwds)

    def test_assumed_standard_repn_behavior(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2])
        m.p = pyo.Param(initialize=0.0)
        expr = m.x[1] + 0 * m.x[2]
        repn = generate_standard_repn(expr)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], m.x[1])
        expr = m.p * m.x[1] + m.x[2]
        repn = generate_standard_repn(expr)
        self.assertEqual(len(repn.linear_vars), 1)
        self.assertIs(repn.linear_vars[0], m.x[2])

    def test_fixed_none_linear_coefficient(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.p = pyo.Param([1, 2], mutable=True, initialize=1.0)
        m.x[3].fix(None)
        expr = 2 * m.x[1] + 3 * m.x[3] * m.p[2] * m.x[2] + m.x[1] ** 2
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet([m.x[1], m.x[2]]))

    def test_incidence_with_mutable_parameter(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.p = pyo.Param(mutable=True, initialize=None)
        expr = m.x[1] + m.p * m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))