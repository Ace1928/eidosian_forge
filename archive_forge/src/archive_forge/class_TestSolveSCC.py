import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.scc_solver import (
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
@unittest.skipUnless(scipy_available, 'SciPy is not available')
@unittest.skipUnless(networkx_available, 'NetworkX is not available')
class TestSolveSCC(unittest.TestCase):

    def test_dynamic_backward_no_solver(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme='BACKWARD')
        time = m.time
        t0 = time.first()
        m.flow_in.fix()
        m.height[t0].fix()
        with self.assertRaisesRegex(RuntimeError, 'An external solver is required*'):
            solve_strongly_connected_components(m)
        for t in time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)
            self.assertFalse(m.flow_out[t].fixed)
            self.assertFalse(m.dhdt[t].fixed)
            self.assertTrue(m.flow_in[t].fixed)

    @unittest.skipUnless(pyo.SolverFactory('ipopt').available(), 'IPOPT is not available')
    def test_dynamic_backward(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme='BACKWARD')
        time = m.time
        t0 = time.first()
        m.flow_in.fix()
        m.height[t0].fix()
        solver = pyo.SolverFactory('ipopt')
        solve_kwds = {'tee': False}
        solve_strongly_connected_components(m, solver=solver, solve_kwds=solve_kwds)
        for con in m.component_data_objects(pyo.Constraint):
            self.assertEqual(pyo.value(con.upper), pyo.value(con.lower))
            self.assertAlmostEqual(pyo.value(con.body), pyo.value(con.upper), delta=1e-07)
        for t in time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)
            self.assertFalse(m.flow_out[t].fixed)
            self.assertFalse(m.dhdt[t].fixed)
            self.assertTrue(m.flow_in[t].fixed)

    def test_dynamic_forward(self):
        nfe = 5
        m = make_dynamic_model(nfe=nfe, scheme='FORWARD')
        time = m.time
        t0 = time.first()
        m.flow_in.fix()
        m.height[t0].fix()
        solve_strongly_connected_components(m)
        for con in m.component_data_objects(pyo.Constraint):
            self.assertEqual(pyo.value(con.upper), pyo.value(con.lower))
            self.assertAlmostEqual(pyo.value(con.body), pyo.value(con.upper), delta=1e-07)
        for t in time:
            if t == t0:
                self.assertTrue(m.height[t].fixed)
            else:
                self.assertFalse(m.height[t].fixed)
            self.assertFalse(m.flow_out[t].fixed)
            self.assertFalse(m.dhdt[t].fixed)
            self.assertTrue(m.flow_in[t].fixed)

    def test_with_calc_var_kwds(self):
        m = pyo.ConcreteModel()
        m.v0 = pyo.Var()
        m.v1 = pyo.Var()
        m.v2 = pyo.Var(initialize=79703634.05074187)
        m.v2.fix()
        m.p0 = pyo.Param(initialize=300000.0)
        m.p1 = pyo.Param(initialize=1296000000000.0)
        m.con0 = pyo.Constraint(expr=m.v0 == m.p0)
        m.con1 = pyo.Constraint(expr=0.0 == m.p1 * m.v1 / m.v0 + m.v2)
        calc_var_kwds = {'eps': 1e-07}
        results = solve_strongly_connected_components(m, calc_var_kwds=calc_var_kwds)
        self.assertEqual(len(results), 2)
        self.assertAlmostEqual(m.v0.value, m.p0.value)
        self.assertAlmostEqual(m.v1.value, -18.4499152895)

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

    def test_with_inequalities(self):
        """Test that we correctly ignore inequalities"""
        m = pyo.ConcreteModel()
        m.x = pyo.Var([1, 2, 3], initialize=1.0)
        m.eq1 = pyo.Constraint(expr=m.x[1] + 2 * m.x[2] + 0 * m.x[3] == 7)
        m.eq2 = pyo.Constraint(expr=m.x[1] + pyo.log(m.x[1]) == 0)
        m.ineq1 = pyo.Constraint(expr=m.x[1] + 2 * m.x[2] + m.x[3] <= 3)
        results = solve_strongly_connected_components(m)
        self.assertAlmostEqual(m.x[1].value, 0.56714329)
        self.assertAlmostEqual(m.x[2].value, 3.21642835)
        self.assertEqual(m.x[3].value, 1.0)