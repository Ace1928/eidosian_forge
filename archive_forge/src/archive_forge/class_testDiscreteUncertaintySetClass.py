import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.common.config import ConfigBlock, ConfigValue
from pyomo.core.base.set_types import NonNegativeIntegers
from pyomo.core.expr import (
from pyomo.contrib.pyros.util import (
from pyomo.contrib.pyros.util import replace_uncertain_bounds_with_constraints
from pyomo.contrib.pyros.util import get_vars_from_component
from pyomo.contrib.pyros.util import identify_objective_functions
from pyomo.common.collections import Bunch
import time
import math
from pyomo.contrib.pyros.util import time_code
from pyomo.contrib.pyros.uncertainty_sets import (
from pyomo.contrib.pyros.master_problem_methods import (
from pyomo.contrib.pyros.solve_data import MasterProblemData, ROSolveResults
from pyomo.common.dependencies import numpy as np, numpy_available
from pyomo.common.dependencies import scipy as sp, scipy_available
from pyomo.environ import maximize as pyo_max
from pyomo.common.errors import ApplicationError
from pyomo.opt import (
from pyomo.environ import (
import logging
from itertools import chain
class testDiscreteUncertaintySetClass(unittest.TestCase):
    """
    Discrete uncertainty sets. Required inputis a scenarios list.
    """

    def test_normal_discrete_set_construction_and_update(self):
        """
        Test DiscreteScenarioSet constructor and setter work normally
        when scenarios are appropriate.
        """
        scenarios = [[0, 0, 0], [1, 2, 3]]
        dset = DiscreteScenarioSet(scenarios)
        np.testing.assert_allclose(scenarios, dset.scenarios, err_msg='BoxSet bounds not as expected')
        new_scenarios = [[0, 1, 2], [1, 2, 0], [3, 5, 4]]
        dset.scenarios = new_scenarios
        np.testing.assert_allclose(new_scenarios, dset.scenarios, err_msg='BoxSet bounds not as expected')

    def test_error_on_discrete_set_dim_change(self):
        """
        Test ValueError raised when attempting to update
        DiscreteScenarioSet dimension.
        """
        scenarios = [[1, 2], [3, 4]]
        dset = DiscreteScenarioSet(scenarios)
        exc_str = '.*must have 2 columns.* to match set dimension \\(provided.*with 3 columns\\)'
        with self.assertRaisesRegex(ValueError, exc_str):
            dset.scenarios = [[1, 2, 3], [4, 5, 6]]

    def test_uncertainty_set_with_correct_params(self):
        """
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        """
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        scenarios = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list((id(u) for u in uncertain_params_in_expr)):
                        uncertain_params_in_expr.append(v)
        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()], msg='Uncertain param Var objects used to construct uncertainty set constraint must be the same uncertain param Var objects in the original model.')

    def test_uncertainty_set_with_incorrect_params(self):
        """
        Case in which the set is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        """
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        scenarios = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if id(v) in [id(u) for u in list(identify_variables(expr=con.expr))]:
                    if id(v) not in list((id(u) for u in vars_in_expr)):
                        vars_in_expr.append(v)
        self.assertEqual(len(vars_in_expr), 0, msg='Uncertainty set constraint contains no Var objects, consists of a not potentially variable expression.')

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        scenarios = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        self.assertTrue(_set.point_in_set([0, 0]), msg='Point is not in the DiscreteScenarioSet.')

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0)
        scenarios = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0)]
        _set = DiscreteScenarioSet(scenarios=scenarios)
        config = Block()
        config.uncertainty_set = _set
        DiscreteScenarioSet.add_bounds_on_uncertain_parameters(model=m, config=config)
        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, 'Bounds not added correctly for DiscreteScenarioSet')
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, 'Bounds not added correctly for DiscreteScenarioSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, 'Bounds not added correctly for DiscreteScenarioSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, 'Bounds not added correctly for DiscreteScenarioSet')

    @unittest.skipUnless(baron_license_is_valid, 'Global NLP solver is not available and licensed.')
    def test_two_stg_model_discrete_set_single_scenario(self):
        """
        Test two-stage model under discrete uncertainty with
        a single scenario.
        """
        m = ConcreteModel()
        m.u1 = Param(initialize=1.125, mutable=True)
        m.u2 = Param(initialize=1, mutable=True)
        m.x1 = Var(initialize=0, bounds=(0, None))
        m.x2 = Var(initialize=0, bounds=(0, None))
        m.x3 = Var(initialize=0, bounds=(None, None))
        m.con1 = Constraint(expr=m.x1 * m.u1 ** 0.5 - m.x2 * m.u1 <= 2)
        m.con2 = Constraint(expr=m.x1 ** 2 - m.x2 ** 2 * m.u1 == m.x3)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u2) ** 2)
        discrete_set = DiscreteScenarioSet(scenarios=[(1.125, 1)])
        pyros_solver = SolverFactory('pyros')
        local_subsolver = SolverFactory('baron')
        global_subsolver = SolverFactory('baron')
        results = pyros_solver.solve(model=m, first_stage_variables=[m.x1], second_stage_variables=[m.x2], uncertain_params=[m.u1, m.u2], uncertainty_set=discrete_set, local_solver=local_subsolver, global_solver=global_subsolver, options={'objective_focus': ObjectiveType.worst_case, 'solve_master_globally': True})
        self.assertEqual(results.pyros_termination_condition, pyrosTerminationCondition.robust_optimal, msg='Did not identify robust optimal solution to problem instance.')
        self.assertEqual(results.iterations, 1, msg='PyROS was unable to solve a singleton discrete set instance  successfully within a single iteration.')

    @unittest.skipUnless(baron_license_is_valid, 'Global NLP solver is not available and licensed.')
    def test_two_stg_model_discrete_set(self):
        """
        Test PyROS successfully solves two-stage model with
        multiple scenarios.
        """
        m = ConcreteModel()
        m.x1 = Var(bounds=(0, 10))
        m.x2 = Var(bounds=(0, 10))
        m.u = Param(mutable=True, initialize=1.125)
        m.con = Constraint(expr=sqrt(m.u) * m.x1 - m.u * m.x2 <= 2)
        m.obj = Objective(expr=(m.x1 - 4) ** 2 + (m.x2 - m.u) ** 2)
        discrete_set = DiscreteScenarioSet(scenarios=[[0.25], [1.125], [2]])
        global_solver = SolverFactory('baron')
        pyros_solver = SolverFactory('pyros')
        res = pyros_solver.solve(model=m, first_stage_variables=[m.x1], second_stage_variables=[m.x2], uncertain_params=[m.u], uncertainty_set=discrete_set, local_solver=global_solver, global_solver=global_solver, decision_rule_order=0, solve_master_globally=True, objective_focus=ObjectiveType.worst_case)
        self.assertEqual(res.pyros_termination_condition, pyrosTerminationCondition.robust_optimal, msg='Failed to solve discrete set multiple scenarios instance to robust optimality')