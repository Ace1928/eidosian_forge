import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet
from pyomo.contrib.incidence_analysis.incidence import (
class _TestIncidence(object):
    """Base class with tests for get_incident_variables that should be
    independent of the method used

    """

    def _get_incident_variables(self, expr):
        raise NotImplementedError('_TestIncidence should not be used directly')

    def test_basic_incidence(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        expr = m.x[1] + m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))

    def test_incidence_with_fixed_variable(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1.0)
        expr = m.x[1] + m.x[1] * m.x[2] + m.x[1] * pyo.exp(m.x[3])
        m.x[2].fix()
        variables = self._get_incident_variables(expr)
        var_set = ComponentSet(variables)
        self.assertEqual(var_set, ComponentSet([m.x[1], m.x[3]]))

    def test_incidence_with_named_expression(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3])
        m.subexpr = pyo.Expression(pyo.Integers)
        m.subexpr[1] = m.x[1] * pyo.exp(m.x[3])
        expr = m.x[1] + m.x[1] * m.x[2] + m.subexpr[1]
        variables = self._get_incident_variables(expr)
        self.assertEqual(ComponentSet(variables), ComponentSet(m.x[:]))