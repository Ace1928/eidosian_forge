from contextlib import redirect_stdout
from io import StringIO
import logging
from math import fabs
from os.path import join, normpath
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import Bunch
from pyomo.common.config import ConfigDict, ConfigValue
from pyomo.common.fileutils import import_file, PYOMO_ROOT_DIR
from pyomo.contrib.appsi.solvers.gurobi import Gurobi
from pyomo.contrib.gdpopt.create_oa_subproblems import (
import pyomo.contrib.gdpopt.tests.common_tests as ct
from pyomo.contrib.gdpopt.util import is_feasible, time_code
from pyomo.contrib.mcpp.pyomo_mcpp import mcpp_available
from pyomo.contrib.gdpopt.solve_discrete_problem import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.gdp.tests import models
from pyomo.opt import TerminationCondition
class TestGDPoptUnit(unittest.TestCase):
    """Real unit tests for GDPopt"""

    @unittest.skipUnless(SolverFactory(mip_solver).available(), 'MIP solver not available')
    def test_solve_discrete_problem_unbounded(self):
        m = ConcreteModel()
        m.GDPopt_utils = Block()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.z = Var()
        m.d = Disjunction(expr=[[m.x + m.y >= 5], [m.x - m.y <= 3]])
        m.o = Objective(expr=m.z)
        m.GDPopt_utils.variable_list = [m.x, m.y, m.z]
        m.GDPopt_utils.disjunct_list = [m.d._autodisjuncts[0], m.d._autodisjuncts[1]]
        TransformationFactory('gdp.bigm').apply_to(m)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            solver = SolverFactory('gdpopt.loa')
            dummy = Block()
            dummy.timing = Bunch()
            with time_code(dummy.timing, 'main', is_main_timer=True):
                tc = solve_MILP_discrete_problem(m.GDPopt_utils, dummy, solver.CONFIG(dict(mip_solver=mip_solver)))
            self.assertIn('Discrete problem was unbounded. Re-solving with arbitrary bound values', output.getvalue().strip())
        self.assertIs(tc, TerminationCondition.unbounded)

    @unittest.skipUnless(SolverFactory(mip_solver).available(), 'MIP solver not available')
    def test_solve_lp(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-5, 5))
        m.c = Constraint(expr=m.x >= 1)
        m.o = Objective(expr=m.x)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            results = SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver)
            self.assertIn('Your model is an LP (linear program).', output.getvalue().strip())
            self.assertAlmostEqual(value(m.o.expr), 1)
            self.assertEqual(results.problem.number_of_binary_variables, 0)
            self.assertEqual(results.problem.number_of_integer_variables, 0)
            self.assertEqual(results.problem.number_of_disjunctions, 0)
            self.assertAlmostEqual(results.problem.lower_bound, 1)
            self.assertAlmostEqual(results.problem.upper_bound, 1)

    @unittest.skipUnless(gurobi_available, 'Gurobi not available')
    def test_solve_nlp(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-5, 5))
        m.c = Constraint(expr=m.x >= 1)
        m.o = Objective(expr=m.x ** 2)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            results = SolverFactory('gdpopt.loa').solve(m, nlp_solver='gurobi')
            self.assertIn('Your model is an NLP (nonlinear program).', output.getvalue().strip())
            self.assertAlmostEqual(value(m.o.expr), 1)
            self.assertEqual(results.problem.number_of_binary_variables, 0)
            self.assertEqual(results.problem.number_of_integer_variables, 0)
            self.assertEqual(results.problem.number_of_disjunctions, 0)
            self.assertAlmostEqual(results.problem.lower_bound, 1)
            self.assertAlmostEqual(results.problem.upper_bound, 1)

    @unittest.skipUnless(SolverFactory(mip_solver).available(), 'MIP solver not available')
    def test_solve_constant_obj(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-5, 5))
        m.c = Constraint(expr=m.x >= 1)
        m.o = Objective(expr=1)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver)
            self.assertIn('Your model is an LP (linear program).', output.getvalue().strip())
            self.assertAlmostEqual(value(m.o.expr), 1)

    @unittest.skipUnless(SolverFactory(nlp_solver).available(), 'NLP solver not available')
    def test_no_objective(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-5, 5))
        m.c = Constraint(expr=m.x ** 2 >= 1)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            SolverFactory('gdpopt.loa').solve(m, nlp_solver=nlp_solver)
            self.assertIn('Model has no active objectives. Adding dummy objective.', output.getvalue().strip())
        self.assertIsNone(m.component('dummy_obj'))

    def test_multiple_objectives(self):
        m = ConcreteModel()
        m.x = Var()
        m.o = Objective(expr=m.x)
        m.o2 = Objective(expr=m.x + 1)
        with self.assertRaisesRegex(ValueError, 'Model has multiple active objectives'):
            SolverFactory('gdpopt.loa').solve(m)

    def test_is_feasible_function(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 3), initialize=2)
        m.c = Constraint(expr=m.x == 2)
        GDP_LOA_Solver = SolverFactory('gdpopt.loa')
        self.assertTrue(is_feasible(m, GDP_LOA_Solver.CONFIG()))
        m.c2 = Constraint(expr=m.x <= 1)
        self.assertFalse(is_feasible(m, GDP_LOA_Solver.CONFIG()))
        m = ConcreteModel()
        m.x = Var(bounds=(0, 3), initialize=2)
        m.c = Constraint(expr=m.x >= 5)
        self.assertFalse(is_feasible(m, GDP_LOA_Solver.CONFIG()))
        m = ConcreteModel()
        m.x = Var(bounds=(3, 3), initialize=2)
        self.assertFalse(is_feasible(m, GDP_LOA_Solver.CONFIG()))
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1), initialize=2)
        self.assertFalse(is_feasible(m, GDP_LOA_Solver.CONFIG()))
        m = ConcreteModel()
        m.x = Var(bounds=(0, 1), initialize=2)
        m.d = Disjunct()
        with self.assertRaisesRegex(NotImplementedError, 'Found active disjunct'):
            is_feasible(m, GDP_LOA_Solver.CONFIG())

    @unittest.skipUnless(gurobi_available, 'Gurobi not available')
    def test_infeasible_or_unbounded_mip_termination(self):
        m = ConcreteModel()
        m.x = Var()
        m.c1 = Constraint(expr=m.x >= 2)
        m.c2 = Constraint(expr=m.x <= 1.9)
        m.obj = Objective(expr=m.x)
        results = SolverFactory('gurobi').solve(m)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasibleOrUnbounded)
        config = ConfigDict()
        config.declare('mip_solver', ConfigValue('gurobi'))
        config.declare('mip_solver_args', ConfigValue({}))
        results, termination_condition = distinguish_mip_infeasible_or_unbounded(m, config)
        self.assertEqual(termination_condition, TerminationCondition.infeasible)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)

    def get_GDP_on_block(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-5, 5))
        m.y = Var(bounds=(-2, 6))
        m.b = Block()
        m.b.disjunction = Disjunction(expr=[[m.x + m.y <= 1, m.y >= 0.5], [m.x == 2, m.y == 4], [m.x ** 2 - m.y <= 3]])
        m.disjunction = Disjunction(expr=[[m.x - m.y <= -2, m.y >= -1], [m.x == 0, m.y >= 0], [m.y ** 2 + m.x <= 3]])
        return m

    def test_gloa_cut_generation_ignores_deactivated_constraints(self):
        m = self.get_GDP_on_block()
        m.b.disjunction.disjuncts[0].indicator_var.fix(True)
        m.b.disjunction.disjuncts[1].indicator_var.fix(False)
        m.b.disjunction.disjuncts[2].indicator_var.fix(False)
        m.b.deactivate()
        m.disjunction.disjuncts[0].indicator_var.fix(False)
        m.disjunction.disjuncts[1].indicator_var.fix(True)
        m.disjunction.disjuncts[2].indicator_var.fix(False)
        add_util_block(m)
        util_block = m._gdpopt_cuts
        add_disjunct_list(util_block)
        add_constraints_by_disjunct(util_block)
        add_global_constraint_list(util_block)
        TransformationFactory('gdp.bigm').apply_to(m)
        config = ConfigDict()
        config.declare('integer_tolerance', ConfigValue(1e-06))
        gloa = SolverFactory('gdpopt.gloa')
        constraints = list(gloa._get_active_untransformed_constraints(util_block, config))
        self.assertEqual(len(constraints), 2)
        c1 = constraints[0]
        c2 = constraints[1]
        self.assertIs(c1.body, m.x)
        self.assertEqual(c1.lower, 0)
        self.assertEqual(c1.upper, 0)
        self.assertIs(c2.body, m.y)
        self.assertEqual(c2.lower, 0)
        self.assertIsNone(c2.upper)

    def test_complain_when_no_algorithm_specified(self):
        m = self.get_GDP_on_block()
        with self.assertRaisesRegex(ValueError, 'No algorithm was specified to the solve method. Please specify an algorithm or use an algorithm-specific solver.'):
            SolverFactory('gdpopt').solve(m)

    @unittest.skipIf(not LOA_solvers_available, 'Required subsolvers %s are not available' % (LOA_solvers,))
    def test_solve_block(self):
        m = ConcreteModel()
        m.b = Block()
        m.b.x = Var(bounds=(-5, 5))
        m.b.y = Var(bounds=(-2, 6))
        m.b.disjunction = Disjunction(expr=[[m.b.x + m.b.y <= 1, m.b.y >= 0.5], [m.b.x == 2, m.b.y == 4], [m.b.x ** 2 - m.b.y <= 3]])
        m.disjunction = Disjunction(expr=[[m.b.x - m.b.y <= -2, m.b.y >= -1], [m.b.x == 0, m.b.y >= 0], [m.b.y ** 2 + m.b.x <= 3]])
        m.b.obj = Objective(expr=m.b.x)
        SolverFactory('gdpopt.ric').solve(m.b, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.b.x), -5)
        self.assertEqual(len(m.component_map(Block)), 1)