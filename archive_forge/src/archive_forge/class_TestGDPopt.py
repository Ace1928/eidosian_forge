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
@unittest.skipIf(not LOA_solvers_available, 'Required subsolvers %s are not available' % (LOA_solvers,))
class TestGDPopt(unittest.TestCase):
    """Tests for the GDPopt solver plugin."""

    def test_infeasible_GDP(self):
        """Test for infeasible GDP."""
        m = models.make_infeasible_gdp_model()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            results = SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
            self.assertIn('Set covering problem is infeasible.', output.getvalue().strip())
        self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)
        self.assertIsNone(m.x.value)
        self.assertIsNone(m.d.disjuncts[0].indicator_var.value)
        self.assertIsNone(m.d.disjuncts[1].indicator_var.value)
        m.o.sense = maximize
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            results = SolverFactory('gdpopt').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver, init_algorithm='no_init', algorithm='LOA')
            self.assertIn('GDPopt exiting--problem is infeasible.', output.getvalue().strip())
        self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)
        self.assertIsNotNone(results.solver.user_time)
        self.assertIsNone(m.x.value)
        self.assertIsNone(m.d.disjuncts[0].indicator_var.value)
        self.assertIsNone(m.d.disjuncts[1].indicator_var.value)

    def test_infeasible_gdp_max_binary(self):
        """Test that max binary initialization catches infeasible GDP too"""
        m = models.make_infeasible_gdp_model()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.DEBUG):
            results = SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver, init_algorithm='max_binary')
            self.assertIn('MILP relaxation for initialization was infeasible. Problem is infeasible.', output.getvalue().strip())
        self.assertEqual(results.solver.termination_condition, TerminationCondition.infeasible)

    def test_unbounded_gdp_minimization(self):
        m = ConcreteModel()
        m.GDPopt_utils = Block()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.z = Var()
        m.d = Disjunction(expr=[[m.x + m.y >= 5], [m.x - m.y <= 3]])
        m.o = Objective(expr=m.z)
        m.GDPopt_utils.variable_list = [m.x, m.y, m.z]
        m.GDPopt_utils.disjunct_list = [m.d._autodisjuncts[0], m.d._autodisjuncts[1]]
        results = SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.unbounded)

    def test_unbounded_gdp_maximization(self):
        m = ConcreteModel()
        m.GDPopt_utils = Block()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.z = Var()
        m.d = Disjunction(expr=[[m.x + m.y <= 5], [m.x - m.y >= 3]])
        m.o = Objective(expr=m.z, sense=maximize)
        m.GDPopt_utils.variable_list = [m.x, m.y, m.z]
        m.GDPopt_utils.disjunct_list = [m.d._autodisjuncts[0], m.d._autodisjuncts[1]]
        results = SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.unbounded)

    @unittest.skipUnless(gurobi_available, 'Gurobi solver not available')
    def test_GDP_nonlinear_objective(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.d = Disjunction(expr=[[m.x + m.y >= 5], [m.x - m.y <= 3]])
        m.o = Objective(expr=m.x ** 2)
        SolverFactory('gdpopt.loa').solve(m, mip_solver='gurobi', nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.o), 0)
        m = ConcreteModel()
        m.x = Var(bounds=(-1, 10))
        m.y = Var(bounds=(2, 3))
        m.d = Disjunction(expr=[[m.x + m.y >= 5], [m.x - m.y <= 3]])
        m.o = Objective(expr=-m.x ** 2, sense=maximize)
        print('second')
        SolverFactory('gdpopt.loa').solve(m, mip_solver='gurobi', nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.o), 0)

    def test_nested_disjunctions_set_covering(self):
        m = models.makeNestedNonlinearModel()
        SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver, init_algorithm='set_covering')
        self.assertAlmostEqual(value(m.x), sqrt(2) / 2)
        self.assertAlmostEqual(value(m.y), sqrt(2) / 2)
        self.assertTrue(value(m.disj.disjuncts[1].indicator_var))
        self.assertFalse(value(m.disj.disjuncts[0].indicator_var))
        self.assertTrue(value(m.d1.indicator_var))
        self.assertFalse(value(m.d2.indicator_var))

    def test_equality_propagation_infeasibility_in_subproblems(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-10, 10))
        m.y = Var(bounds=(-10, 10))
        m.disj = Disjunction(expr=[[m.x == m.y, m.y == 2], [m.y == 8], [m.x + m.y >= 4, m.y == m.x + 1]])
        m.cons = Constraint(expr=m.x == 3)
        m.obj = Objective(expr=m.x + m.y)
        SolverFactory('gdpopt').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver, init_algorithm='set_covering', algorithm='RIC')
        self.assertAlmostEqual(value(m.x), 3)
        self.assertAlmostEqual(value(m.y), 4)
        self.assertFalse(value(m.disj.disjuncts[0].indicator_var))
        self.assertFalse(value(m.disj.disjuncts[1].indicator_var))
        self.assertTrue(value(m.disj.disjuncts[2].indicator_var))

    def test_bound_infeasibility_in_subproblems(self):
        m = ConcreteModel()
        m.x = Var(bounds=(2, 4))
        m.y = Var(bounds=(5, 10))
        m.disj = Disjunction(expr=[[m.x == m.y, m.x + m.y >= 8], [m.x == 4]])
        m.obj = Objective(expr=m.x + m.y)
        SolverFactory('gdpopt.ric').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver, init_algorithm='set_covering', tee=True)
        self.assertAlmostEqual(value(m.x), 4)
        self.assertAlmostEqual(value(m.y), 5)
        self.assertFalse(value(m.disj.disjuncts[0].indicator_var))
        self.assertTrue(value(m.disj.disjuncts[1].indicator_var))

    def test_subproblem_preprocessing_encounters_trivial_constraints(self):
        m = ConcreteModel()
        m.x = Var(bounds=(0, 10))
        m.z = Var(bounds=(-10, 10))
        m.disjunction = Disjunction(expr=[[m.x == 0, m.z >= 4], [m.x + m.z <= 0]])
        m.cons = Constraint(expr=m.x * m.z <= 0)
        m.obj = Objective(expr=-m.z)
        m.disjunction.disjuncts[0].indicator_var.fix(True)
        m.disjunction.disjuncts[1].indicator_var.fix(False)
        SolverFactory('gdpopt.ric').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver, init_algorithm='fix_disjuncts')
        self.assertEqual(value(m.x), 0)
        self.assertEqual(value(m.z), 10)
        self.assertTrue(value(m.disjunction.disjuncts[0].indicator_var))
        self.assertFalse(value(m.disjunction.disjuncts[1].indicator_var))

    def make_convex_circle_and_circle_slice_disjunction(self):
        m = ConcreteModel()
        m.x = Var(bounds=(-10, 18))
        m.y = Var(bounds=(0, 7))
        m.obj = Objective(expr=m.x ** 2 + m.y)
        m.disjunction = Disjunction(expr=[[m.x ** 2 + m.y ** 2 <= 3, m.y >= 1], (m.x - 3) ** 2 + (m.y - 2) ** 2 <= 1])
        return m

    def test_some_vars_only_in_subproblem(self):
        m = self.make_convex_circle_and_circle_slice_disjunction()
        results = SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(results.problem.upper_bound, 1)
        self.assertAlmostEqual(value(m.x), 0)
        self.assertAlmostEqual(value(m.y), 1)

    def test_fixed_vars_honored(self):
        m = self.make_convex_circle_and_circle_slice_disjunction()
        m.disjunction.disjuncts[0].indicator_var.fix(False)
        SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertTrue(value(m.disjunction.disjuncts[1].indicator_var))
        self.assertFalse(value(m.disjunction.disjuncts[0].indicator_var))
        self.assertTrue(m.disjunction.disjuncts[0].indicator_var.fixed)
        self.assertAlmostEqual(value(m.x), 2.029, places=3)
        self.assertAlmostEqual(value(m.y), 1.761, places=3)
        self.assertAlmostEqual(value(m.obj), 5.878, places=3)
        m.disjunction.disjuncts[0].indicator_var.fixed = False
        m.x.fix(3)
        SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertTrue(value(m.disjunction.disjuncts[1].indicator_var))
        self.assertFalse(value(m.disjunction.disjuncts[0].indicator_var))
        self.assertEqual(value(m.x), 3)
        self.assertTrue(m.x.fixed)
        self.assertAlmostEqual(value(m.y), 1)
        self.assertAlmostEqual(value(m.obj), 10)

    def test_ignore_set_for_oa_cuts(self):
        m = self.make_convex_circle_and_circle_slice_disjunction()
        m.disjunction.disjuncts[1].GDPopt_ignore_OA = [m.disjunction.disjuncts[1].constraint[1]]
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.DEBUG):
            SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
            self.assertIn('OA cut addition for disjunction_disjuncts[1].constraint[1] skipped because it is in the ignore set.', output.getvalue().strip())
        self.assertAlmostEqual(value(m.x), 0)
        self.assertAlmostEqual(value(m.y), 1)

    def test_reverse_numeric_differentiation_in_LOA(self):
        m = ConcreteModel()
        m.s = RangeSet(1300)
        m.x = Var(m.s, bounds=(-10, 10))
        m.d1 = Disjunct()
        m.d1.hypersphere = Constraint(expr=sum((m.x[i] ** 2 for i in m.s)) <= 1)
        m.d2 = Disjunct()
        m.d2.translated_hyper_sphere = Constraint(expr=sum(((m.x[i] - i) ** 2 for i in m.s)) <= 1)
        m.disjunction = Disjunction(expr=[m.d1, m.d2])
        m.obj = Objective(expr=sum((m.x[i] for i in m.s)))
        results = SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
        self.assertTrue(value(m.d1.indicator_var))
        self.assertFalse(value(m.d2.indicator_var))
        x_val = -sqrt(1300) / 1300
        for x in m.x.values():
            self.assertAlmostEqual(value(x), x_val)
        self.assertAlmostEqual(results.problem.upper_bound, 1300 * x_val, places=6)

    def test_logical_constraints_on_disjuncts(self):
        m = models.makeLogicalConstraintsOnDisjuncts()
        SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    def test_logical_constraints_on_disjuncts_nonlinear_convex(self):
        m = models.makeLogicalConstraintsOnDisjuncts_NonlinearConvex()
        results = SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertEqual(results.solver.termination_condition, TerminationCondition.optimal)
        self.assertAlmostEqual(value(m.x), 4)

    def test_nested_disjunctions_no_init(self):
        m = models.makeNestedNonlinearModel()
        SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver, init_algorithm='no_init')
        self.assertAlmostEqual(value(m.x), sqrt(2) / 2)
        self.assertAlmostEqual(value(m.y), sqrt(2) / 2)

    def test_nested_disjunctions_max_binary(self):
        m = models.makeNestedNonlinearModel()
        SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver, init_algorithm='max_binary')
        self.assertAlmostEqual(value(m.x), sqrt(2) / 2)
        self.assertAlmostEqual(value(m.y), sqrt(2) / 2)

    def test_boolean_vars_on_disjuncts(self):
        m = models.makeBooleanVarsOnDisjuncts()
        SolverFactory('gdpopt.loa').solve(m, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertAlmostEqual(value(m.x), 8)

    def test_LOA_8PP_default_init(self):
        """Test logic-based outer approximation with 8PP."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.loa').solve(eight_process, mip_solver=mip_solver, nlp_solver=nlp_solver)
        ct.check_8PP_solution(self, eight_process, results)

    def test_iteration_limit(self):
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            results = SolverFactory('gdpopt.loa').solve(eight_process, mip_solver=mip_solver, nlp_solver=nlp_solver, iterlim=2)
            self.assertIn('GDPopt unable to converge bounds within iteration limit of 2 iterations.', output.getvalue().strip())
        self.assertEqual(results.solver.termination_condition, TerminationCondition.maxIterations)

    def test_time_limit(self):
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.INFO):
            results = SolverFactory('gdpopt.loa').solve(eight_process, mip_solver=mip_solver, nlp_solver=nlp_solver, time_limit=1)
            self.assertIn('GDPopt exiting--Did not converge bounds before time limit of 1 seconds.', output.getvalue().strip())
        self.assertEqual(results.solver.termination_condition, TerminationCondition.maxTimeLimit)

    def test_LOA_8PP_logical_default_init(self):
        """Test logic-based outer approximation with 8PP."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_logical.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.loa').solve(eight_process, mip_solver=mip_solver, nlp_solver=nlp_solver, tee=False)
        ct.check_8PP_logical_solution(self, eight_process, results)

    @unittest.skipUnless(SolverFactory('gams').available(exception_flag=False), 'GAMS solver not available')
    def test_LOA_8PP_gams_solver(self):
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.loa').solve(eight_process, mip_solver=mip_solver, nlp_solver='gams', max_slack=0, tee=False)
        ct.check_8PP_solution(self, eight_process, results)

    def test_LOA_8PP_force_NLP(self):
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.loa').solve(eight_process, mip_solver=mip_solver, nlp_solver=nlp_solver, force_subproblem_nlp=True, tee=False)
        ct.check_8PP_solution(self, eight_process, results)

    def test_LOA_strip_pack_default_init(self):
        """Test logic-based outer approximation with strip packing."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt.loa').solve(strip_pack, mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 11) <= 0.01)

    def test_LOA_strip_pack_logical_constraints(self):
        """Test logic-based outer approximation with variation of strip
        packing with some logical constraints."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        strip_pack.Rec3AboveOrBelowRec1 = LogicalConstraint(expr=strip_pack.no_overlap[1, 3].disjuncts[2].indicator_var.lor(strip_pack.no_overlap[1, 3].disjuncts[3].indicator_var))
        strip_pack.Rec3RightOrLeftOfRec2 = LogicalConstraint(expr=strip_pack.no_overlap[2, 3].disjuncts[0].indicator_var.lor(strip_pack.no_overlap[2, 3].disjuncts[1].indicator_var))
        SolverFactory('gdpopt.loa').solve(strip_pack, mip_solver=mip_solver, nlp_solver=nlp_solver, subproblem_presolve=False)
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 13) <= 0.01)

    @unittest.pytest.mark.expensive
    def test_LOA_constrained_layout_default_init(self):
        """Test LOA with constrained layout."""
        exfile = import_file(join(exdir, 'constrained_layout', 'cons_layout_model.py'))
        cons_layout = exfile.build_constrained_layout_model()
        SolverFactory('gdpopt.loa').solve(cons_layout, mip_solver=mip_solver, nlp_solver=nlp_solver, iterlim=120, max_slack=5)
        objective_value = value(cons_layout.min_dist_cost.expr)
        self.assertTrue(fabs(objective_value - 41573) <= 200, 'Objective value of %s instead of 41573' % objective_value)

    def test_LOA_8PP_maxBinary(self):
        """Test logic-based OA with max_binary initialization."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.loa').solve(eight_process, init_algorithm='max_binary', mip_solver=mip_solver, nlp_solver=nlp_solver)
        ct.check_8PP_solution(self, eight_process, results)

    def test_LOA_8PP_logical_maxBinary(self):
        """Test logic-based OA with max_binary initialization."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_logical.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.loa').solve(eight_process, init_algorithm='max_binary', mip_solver=mip_solver, nlp_solver=nlp_solver)
        ct.check_8PP_logical_solution(self, eight_process, results)

    def test_LOA_strip_pack_maxBinary(self):
        """Test LOA with strip packing using max_binary initialization."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        SolverFactory('gdpopt.loa').solve(strip_pack, init_algorithm='max_binary', mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 11) <= 0.01)

    def test_LOA_strip_pack_maxBinary_logical_constraints(self):
        """Test LOA with strip packing using max_binary initialization and
        logical constraints."""
        exfile = import_file(join(exdir, 'strip_packing', 'strip_packing_concrete.py'))
        strip_pack = exfile.build_rect_strip_packing_model()
        strip_pack.Rec3AboveOrBelowRec1 = LogicalConstraint(expr=strip_pack.no_overlap[1, 3].disjuncts[2].indicator_var.lor(strip_pack.no_overlap[1, 3].disjuncts[3].indicator_var))
        strip_pack.Rec3RightOrLeftOfRec2 = LogicalConstraint(expr=strip_pack.no_overlap[2, 3].disjuncts[0].indicator_var.lor(strip_pack.no_overlap[2, 3].disjuncts[1].indicator_var))
        SolverFactory('gdpopt.loa').solve(strip_pack, init_algorithm='max_binary', mip_solver=mip_solver, nlp_solver=nlp_solver)
        self.assertTrue(fabs(value(strip_pack.total_length.expr) - 13) <= 0.01)

    def test_LOA_8PP_fixed_disjuncts(self):
        """Test LOA with 8PP using fixed disjuncts initialization."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        initialize = [eight_process.use_unit_1or2.disjuncts[0], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[1], eight_process.use_unit_8ornot.disjuncts[0]]
        for disj in eight_process.component_data_objects(Disjunct):
            if disj in initialize:
                disj.binary_indicator_var.set_value(1)
            else:
                disj.binary_indicator_var.set_value(0)
        results = SolverFactory('gdpopt.loa').solve(eight_process, init_algorithm='fix_disjuncts', mip_solver=mip_solver, nlp_solver=nlp_solver)
        ct.check_8PP_solution(self, eight_process, results)

    def test_LOA_custom_disjuncts_with_silly_components_in_list(self):
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        eight_process.goofy = Disjunct()
        eight_process.goofy.deactivate()
        initialize = [[eight_process.use_unit_1or2.disjuncts[0], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[1], eight_process.use_unit_8ornot.disjuncts[0], eight_process.goofy], [eight_process.use_unit_1or2.disjuncts[1], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[0], eight_process.use_unit_8ornot.disjuncts[0]]]
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.gdpopt', logging.WARNING):
            results = SolverFactory('gdpopt.loa').solve(eight_process, init_algorithm='custom_disjuncts', custom_init_disjuncts=initialize, mip_solver=mip_solver, nlp_solver=nlp_solver)
            self.assertIn('The following disjuncts from the custom disjunct initialization set number 0 were unused: goofy', output.getvalue().strip())
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 0.01)

    def test_LOA_custom_disjuncts(self):
        """Test logic-based OA with custom disjuncts initialization."""
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        eight_process = exfile.build_eight_process_flowsheet()
        initialize = [[eight_process.use_unit_1or2.disjuncts[0], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[1], eight_process.use_unit_8ornot.disjuncts[0]], [eight_process.use_unit_1or2.disjuncts[1], eight_process.use_unit_3ornot.disjuncts[1], eight_process.use_unit_4or5ornot.disjuncts[0], eight_process.use_unit_6or7ornot.disjuncts[0], eight_process.use_unit_8ornot.disjuncts[0]]]

        def assert_correct_disjuncts_active(solver, subprob_util_block, discrete_problem_util_block):
            iteration = solver.initialization_iteration
            discrete_problem = discrete_problem_util_block.model()
            subprob = subprob_util_block.model()
            if iteration >= 2:
                return
            disjs_should_be_active = initialize[iteration]
            seen = set()
            for orig_disj in disjs_should_be_active:
                parent_nm = orig_disj.parent_component().name
                idx = orig_disj.index()
                discrete_problem_parent = discrete_problem.component(parent_nm)
                subprob_parent = subprob.component(parent_nm)
                self.assertIsInstance(discrete_problem_parent, Disjunct)
                self.assertIsInstance(subprob_parent, Block)
                discrete_problem_disj = discrete_problem_parent[idx]
                subprob_disj = subprob_parent[idx]
                self.assertTrue(value(discrete_problem_disj.indicator_var))
                self.assertTrue(subprob_disj.active)
                seen.add(subprob_disj)
            for disj in subprob_util_block.disjunct_list:
                if disj not in seen:
                    self.assertFalse(disj.active)
        SolverFactory('gdpopt.loa').solve(eight_process, init_algorithm='custom_disjuncts', custom_init_disjuncts=initialize, mip_solver=mip_solver, nlp_solver=nlp_solver, subproblem_initialization_method=assert_correct_disjuncts_active)
        self.assertTrue(fabs(value(eight_process.profit.expr) - 68) <= 0.01)

    @unittest.skipUnless(Gurobi().available(), 'APPSI Gurobi solver is not available')
    def test_auto_persistent_solver(self):
        exfile = import_file(join(exdir, 'eight_process', 'eight_proc_model.py'))
        m = exfile.build_eight_process_flowsheet()
        results = SolverFactory('gdpopt.loa').solve(m, mip_solver='appsi_gurobi')
        self.assertTrue(fabs(value(m.profit.expr) - 68) <= 0.01)
        ct.check_8PP_solution(self, m, results)