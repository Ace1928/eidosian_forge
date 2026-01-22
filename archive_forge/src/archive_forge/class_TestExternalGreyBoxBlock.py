import itertools
import pyomo.common.unittest as unittest
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.core.expr.visitor import identify_variables
import pyomo.environ as pyo
from pyomo.contrib.pynumero.dependencies import (
from pyomo.common.dependencies.scipy import sparse as sps
from pyomo.contrib.pynumero.asl import AmplInterface
from pyomo.contrib.pynumero.algorithms.solvers.cyipopt_solver import cyipopt_available
from pyomo.contrib.pynumero.algorithms.solvers.implicit_functions import (
from pyomo.contrib.pynumero.interfaces.external_pyomo_model import (
from pyomo.contrib.pynumero.interfaces.external_grey_box import (
from pyomo.contrib.pynumero.interfaces.pyomo_grey_box_nlp import (
from pyomo.contrib.pynumero.interfaces.tests.external_grey_box_models import (
class TestExternalGreyBoxBlock(unittest.TestCase):

    def test_construct_scalar(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block
        self.assertIs(type(block), ScalarExternalGreyBoxBlock)
        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
        block.set_external_model(ex_model)
        self.assertEqual(len(block.inputs), len(input_vars))
        self.assertEqual(len(block.outputs), 0)
        self.assertEqual(len(block._equality_constraint_names), 2)

    def test_construct_indexed(self):
        block = ExternalGreyBoxBlock([0, 1, 2], concrete=True)
        self.assertIs(type(block), IndexedExternalGreyBoxBlock)
        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
        for i in block:
            b = block[i]
            b.set_external_model(ex_model)
            self.assertEqual(len(b.inputs), len(input_vars))
            self.assertEqual(len(b.outputs), 0)
            self.assertEqual(len(b._equality_constraint_names), 2)

    @unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
    def test_solve_square(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block
        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
        block.set_external_model(ex_model)
        _add_linking_constraints(m)
        m.a.fix(1)
        m.b.fix(2)
        m.r.fix(3)
        m.obj = pyo.Objective(expr=0)
        solver = pyo.SolverFactory('cyipopt')
        solver.solve(m)
        self.assertFalse(m_ex.a.fixed)
        self.assertFalse(m_ex.b.fixed)
        self.assertFalse(m_ex.r.fixed)
        m_ex.a.fix(1)
        m_ex.b.fix(2)
        m_ex.r.fix(3)
        ipopt = pyo.SolverFactory('ipopt')
        ipopt.solve(m_ex)
        x = m.ex_block.inputs['input_3']
        y = m.ex_block.inputs['input_4']
        self.assertAlmostEqual(m_ex.x.value, x.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.y.value, y.value, delta=1e-08)

    @unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
    def test_optimize(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block
        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
        block.set_external_model(ex_model)
        a = m.ex_block.inputs['input_0']
        b = m.ex_block.inputs['input_1']
        r = m.ex_block.inputs['input_2']
        x = m.ex_block.inputs['input_3']
        y = m.ex_block.inputs['input_4']
        m.obj = pyo.Objective(expr=(x - 2.0) ** 2 + (y - 2.0) ** 2 + (a - 2.0) ** 2 + (b - 2.0) ** 2 + (r - 2.0) ** 2)
        solver = pyo.SolverFactory('cyipopt')
        solver.solve(m)
        m_ex.obj = pyo.Objective(expr=(m_ex.x - 2.0) ** 2 + (m_ex.y - 2.0) ** 2 + (m_ex.a - 2.0) ** 2 + (m_ex.b - 2.0) ** 2 + (m_ex.r - 2.0) ** 2)
        m_ex.a.set_value(0.0)
        m_ex.b.set_value(0.0)
        m_ex.r.set_value(0.0)
        m_ex.y.set_value(0.0)
        m_ex.x.set_value(0.0)
        ipopt = pyo.SolverFactory('ipopt')
        ipopt.solve(m_ex)
        self.assertAlmostEqual(m_ex.a.value, a.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.b.value, b.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.r.value, r.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.x.value, x.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.y.value, y.value, delta=1e-08)

    @unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
    def test_optimize_with_cyipopt_for_inner_problem(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block
        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        solver_options = dict(solver_class=CyIpoptSolverWrapper)
        ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons, solver_options=solver_options)
        block.set_external_model(ex_model)
        a = m.ex_block.inputs['input_0']
        b = m.ex_block.inputs['input_1']
        r = m.ex_block.inputs['input_2']
        x = m.ex_block.inputs['input_3']
        y = m.ex_block.inputs['input_4']
        m.obj = pyo.Objective(expr=(x - 2.0) ** 2 + (y - 2.0) ** 2 + (a - 2.0) ** 2 + (b - 2.0) ** 2 + (r - 2.0) ** 2)
        solver = pyo.SolverFactory('cyipopt')
        solver.solve(m)
        m_ex.obj = pyo.Objective(expr=(m_ex.x - 2.0) ** 2 + (m_ex.y - 2.0) ** 2 + (m_ex.a - 2.0) ** 2 + (m_ex.b - 2.0) ** 2 + (m_ex.r - 2.0) ** 2)
        m_ex.a.set_value(0.0)
        m_ex.b.set_value(0.0)
        m_ex.r.set_value(0.0)
        m_ex.y.set_value(0.0)
        m_ex.x.set_value(0.0)
        ipopt = pyo.SolverFactory('ipopt')
        ipopt.solve(m_ex)
        self.assertAlmostEqual(m_ex.a.value, a.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.b.value, b.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.r.value, r.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.x.value, x.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.y.value, y.value, delta=1e-08)

    @unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
    def test_optimize_no_decomposition(self):
        m = pyo.ConcreteModel()
        m.ex_block = ExternalGreyBoxBlock(concrete=True)
        block = m.ex_block
        m_ex = _make_external_model()
        input_vars = [m_ex.a, m_ex.b, m_ex.r, m_ex.x_out, m_ex.y_out]
        external_vars = [m_ex.x, m_ex.y]
        residual_cons = [m_ex.c_out_1, m_ex.c_out_2]
        external_cons = [m_ex.c_ex_1, m_ex.c_ex_2]
        ex_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons, solver_class=ImplicitFunctionSolver)
        block.set_external_model(ex_model)
        a = m.ex_block.inputs['input_0']
        b = m.ex_block.inputs['input_1']
        r = m.ex_block.inputs['input_2']
        x = m.ex_block.inputs['input_3']
        y = m.ex_block.inputs['input_4']
        m.obj = pyo.Objective(expr=(x - 2.0) ** 2 + (y - 2.0) ** 2 + (a - 2.0) ** 2 + (b - 2.0) ** 2 + (r - 2.0) ** 2)
        solver = pyo.SolverFactory('cyipopt')
        solver.solve(m)
        m_ex.obj = pyo.Objective(expr=(m_ex.x - 2.0) ** 2 + (m_ex.y - 2.0) ** 2 + (m_ex.a - 2.0) ** 2 + (m_ex.b - 2.0) ** 2 + (m_ex.r - 2.0) ** 2)
        m_ex.a.set_value(0.0)
        m_ex.b.set_value(0.0)
        m_ex.r.set_value(0.0)
        m_ex.y.set_value(0.0)
        m_ex.x.set_value(0.0)
        ipopt = pyo.SolverFactory('ipopt')
        ipopt.solve(m_ex)
        self.assertAlmostEqual(m_ex.a.value, a.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.b.value, b.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.r.value, r.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.x.value, x.value, delta=1e-08)
        self.assertAlmostEqual(m_ex.y.value, y.value, delta=1e-08)

    def test_construct_dynamic(self):
        m = make_dynamic_model()
        time = m.time
        t0 = m.time.first()
        inputs = [m.h, m.dhdt, m.flow_in]
        ext_vars = [m.flow_out]
        residuals = [m.h_diff_eqn]
        ext_cons = [m.flow_out_eqn]
        external_model_dict = {t: ExternalPyomoModel([var[t] for var in inputs], [var[t] for var in ext_vars], [con[t] for con in residuals], [con[t] for con in ext_cons]) for t in time}
        reduced_space = pyo.Block(concrete=True)
        reduced_space.external_block = ExternalGreyBoxBlock(time, external_model=external_model_dict)
        block = reduced_space.external_block
        block[t0].deactivate()
        self.assertIs(type(block), IndexedExternalGreyBoxBlock)
        for t in time:
            b = block[t]
            self.assertEqual(len(b.inputs), len(inputs))
            self.assertEqual(len(b.outputs), 0)
            self.assertEqual(len(b._equality_constraint_names), len(residuals))
        reduced_space.diff_var = pyo.Reference(m.h)
        reduced_space.deriv_var = pyo.Reference(m.dhdt)
        reduced_space.input_var = pyo.Reference(m.flow_in)
        reduced_space.disc_eqn = pyo.Reference(m.dhdt_disc_eqn)
        pyomo_vars = list(reduced_space.component_data_objects(pyo.Var))
        pyomo_cons = list(reduced_space.component_data_objects(pyo.Constraint))
        self.assertEqual(len(pyomo_vars), len(inputs) * len(time))
        self.assertEqual(len(pyomo_cons), len(time) - 1)
        reduced_space._obj = pyo.Objective(expr=0)
        block[:].inputs[:].set_value(1.0)
        reduced_space.const_input_eqn = pyo.Constraint(expr=reduced_space.input_var[2] - reduced_space.input_var[1] == 0)
        nlp = PyomoNLPWithGreyBoxBlocks(reduced_space)
        self.assertEqual(nlp.n_primals(), (2 + len(inputs)) * (len(time) - 1) + len(time))
        self.assertEqual(nlp.n_constraints(), (len(residuals) + 1) * (len(time) - 1) + 1)

    @unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
    def test_solve_square_dynamic(self):
        m = make_dynamic_model()
        time = m.time
        t0 = m.time.first()
        m.h[t0].fix(1.2)
        m.flow_in.fix(1.5)
        reduced_space = pyo.Block(concrete=True)
        reduced_space.diff_var = pyo.Reference(m.h)
        reduced_space.deriv_var = pyo.Reference(m.dhdt)
        reduced_space.input_var = pyo.Reference(m.flow_in)
        reduced_space.disc_eq = pyo.Reference(m.dhdt_disc_eqn)
        reduced_space.external_block = ExternalGreyBoxBlock(time)
        block = reduced_space.external_block
        block[t0].deactivate()
        for t in time:
            if t != t0:
                input_vars = [m.h[t], m.dhdt[t]]
                external_vars = [m.flow_out[t]]
                residual_cons = [m.h_diff_eqn[t]]
                external_cons = [m.flow_out_eqn[t]]
                external_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
                block[t].set_external_model(external_model)
        n_inputs = len(input_vars)

        def linking_constraint_rule(m, i, t):
            if t == t0:
                return pyo.Constraint.Skip
            if i == 0:
                return m.diff_var[t] == m.external_block[t].inputs['input_0']
            elif i == 1:
                return m.deriv_var[t] == m.external_block[t].inputs['input_1']
        reduced_space.linking_constraint = pyo.Constraint(range(n_inputs), time, rule=linking_constraint_rule)
        for t in time:
            if t != t0:
                block[t].inputs['input_0'].set_value(m.h[t].value)
                block[t].inputs['input_1'].set_value(m.dhdt[t].value)
        reduced_space._obj = pyo.Objective(expr=0)
        solver = pyo.SolverFactory('cyipopt')
        results = solver.solve(reduced_space, tee=True)
        h_target = [1.2, 0.852923, 0.690725]
        dhdt_target = [-0.69089, -0.347077, -0.162198]
        flow_out_target = [2.19098, 1.847077, 1.662198]
        for t in time:
            if t == t0:
                continue
            values = [m.h[t].value, m.dhdt[t].value, m.flow_out[t].value]
            target_values = [h_target[t], dhdt_target[t], flow_out_target[t]]
            self.assertStructuredAlmostEqual(values, target_values, delta=1e-05)

    @unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
    def test_optimize_dynamic(self):
        m = make_dynamic_model()
        time = m.time
        t0 = m.time.first()
        m.h[t0].fix(1.2)
        m.flow_in[t0].fix(1.5)
        m.obj = pyo.Objective(expr=sum(((m.h[t] - 2.0) ** 2 for t in m.time if t != t0)))
        reduced_space = pyo.Block(concrete=True)
        reduced_space.diff_var = pyo.Reference(m.h)
        reduced_space.deriv_var = pyo.Reference(m.dhdt)
        reduced_space.input_var = pyo.Reference(m.flow_in)
        reduced_space.disc_eq = pyo.Reference(m.dhdt_disc_eqn)
        reduced_space.objective = pyo.Reference(m.obj)
        reduced_space.external_block = ExternalGreyBoxBlock(time)
        block = reduced_space.external_block
        block[t0].deactivate()
        for t in time:
            if t != t0:
                input_vars = [m.h[t], m.dhdt[t], m.flow_in[t]]
                external_vars = [m.flow_out[t]]
                residual_cons = [m.h_diff_eqn[t]]
                external_cons = [m.flow_out_eqn[t]]
                external_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
                block[t].set_external_model(external_model)
        n_inputs = len(input_vars)

        def linking_constraint_rule(m, i, t):
            if t == t0:
                return pyo.Constraint.Skip
            if i == 0:
                return m.diff_var[t] == m.external_block[t].inputs['input_0']
            elif i == 1:
                return m.deriv_var[t] == m.external_block[t].inputs['input_1']
            elif i == 2:
                return m.input_var[t] == m.external_block[t].inputs['input_2']
        reduced_space.linking_constraint = pyo.Constraint(range(n_inputs), time, rule=linking_constraint_rule)
        for t in time:
            if t != t0:
                block[t].inputs['input_0'].set_value(m.h[t].value)
                block[t].inputs['input_1'].set_value(m.dhdt[t].value)
                block[t].inputs['input_2'].set_value(m.flow_in[t].value)
        solver = pyo.SolverFactory('cyipopt')
        results = solver.solve(reduced_space)
        h_target = [1.2, 2.0, 2.0]
        dhdt_target = [-0.69089, 0.8, 0.0]
        flow_in_target = [1.5, 3.628427, 2.828427]
        flow_out_target = [2.19089, 2.828427, 2.828427]
        for t in time:
            if t == t0:
                continue
            values = [m.h[t].value, m.dhdt[t].value, m.flow_out[t].value, m.flow_in[t].value]
            target_values = [h_target[t], dhdt_target[t], flow_out_target[t], flow_in_target[t]]
            self.assertStructuredAlmostEqual(values, target_values, delta=1e-05)

    @unittest.skipUnless(cyipopt_available, 'cyipopt is not available')
    def test_optimize_dynamic_references(self):
        """
        When when pre-existing variables are attached to the EGBB
        as references, linking constraints are no longer necessary.
        """
        m = make_dynamic_model()
        time = m.time
        t0 = m.time.first()
        m.h[t0].fix(1.2)
        m.flow_in[t0].fix(1.5)
        m.obj = pyo.Objective(expr=sum(((m.h[t] - 2.0) ** 2 for t in m.time if t != t0)))
        reduced_space = pyo.Block(concrete=True)
        reduced_space.diff_var = pyo.Reference(m.h)
        reduced_space.deriv_var = pyo.Reference(m.dhdt)
        reduced_space.input_var = pyo.Reference(m.flow_in)
        reduced_space.disc_eq = pyo.Reference(m.dhdt_disc_eqn)
        reduced_space.objective = pyo.Reference(m.obj)
        reduced_space.external_block = ExternalGreyBoxBlock(time)
        block = reduced_space.external_block
        block[t0].deactivate()
        for t in time:
            if t != t0:
                input_vars = [m.h[t], m.dhdt[t], m.flow_in[t]]
                external_vars = [m.flow_out[t]]
                residual_cons = [m.h_diff_eqn[t]]
                external_cons = [m.flow_out_eqn[t]]
                external_model = ExternalPyomoModel(input_vars, external_vars, residual_cons, external_cons)
                block[t].set_external_model(external_model, inputs=input_vars)
        solver = pyo.SolverFactory('cyipopt')
        results = solver.solve(reduced_space)
        h_target = [1.2, 2.0, 2.0]
        dhdt_target = [-0.69089, 0.8, 0.0]
        flow_in_target = [1.5, 3.628427, 2.828427]
        flow_out_target = [2.19089, 2.828427, 2.828427]
        for t in time:
            if t == t0:
                continue
            values = [m.h[t].value, m.dhdt[t].value, m.flow_out[t].value, m.flow_in[t].value]
            target_values = [h_target[t], dhdt_target[t], flow_out_target[t], flow_in_target[t]]
            self.assertStructuredAlmostEqual(values, target_values, delta=1e-05)