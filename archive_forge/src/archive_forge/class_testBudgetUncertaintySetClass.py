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
class testBudgetUncertaintySetClass(unittest.TestCase):
    """
    Budget uncertainty sets.
    Required inputs are matrix budget_membership_mat, rhs_vec.
    """

    def test_normal_budget_construction_and_update(self):
        """
        Test BudgetSet constructor and attribute setters work
        appropriately.
        """
        budget_mat = [[1, 0, 1], [0, 1, 0]]
        budget_rhs_vec = [1, 3]
        buset = BudgetSet(budget_mat, budget_rhs_vec)
        np.testing.assert_allclose(budget_mat, buset.budget_membership_mat)
        np.testing.assert_allclose(budget_rhs_vec, buset.budget_rhs_vec)
        np.testing.assert_allclose([[1, 0, 1], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], buset.coefficients_mat)
        np.testing.assert_allclose([1, 3, 0, 0, 0], buset.rhs_vec)
        np.testing.assert_allclose(np.zeros(3), buset.origin)
        buset.budget_membership_mat = [[1, 1, 0], [0, 0, 1]]
        buset.budget_rhs_vec = [3, 4]
        np.testing.assert_allclose([[1, 1, 0], [0, 0, 1]], buset.budget_membership_mat)
        np.testing.assert_allclose([3, 4], buset.budget_rhs_vec)
        np.testing.assert_allclose([[1, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], buset.coefficients_mat)
        np.testing.assert_allclose([3, 4, 0, 0, 0], buset.rhs_vec)
        buset.origin = [1, 0, -1.5]
        np.testing.assert_allclose([1, 0, -1.5], buset.origin)

    def test_error_on_budget_set_dim_change(self):
        """
        BudgetSet dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        budget set dimension.
        """
        budget_mat = [[1, 0, 1], [0, 1, 0]]
        budget_rhs_vec = [1, 3]
        bu_set = BudgetSet(budget_mat, budget_rhs_vec)
        exc_str = '.*must have 3 columns to match set dimension \\(provided.*1 columns\\)'
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.budget_membership_mat = [[1], [1]]
        exc_str = '.*must have 3 entries to match set dimension \\(provided.*4 entries\\)'
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.origin = [1, 2, 1, 0]

    def test_error_on_budget_member_mat_row_change(self):
        """
        Number of rows of budget membership mat is immutable.
        Hence, size of budget_rhs_vec is also immutable.
        """
        budget_mat = [[1, 0, 1], [0, 1, 0]]
        budget_rhs_vec = [1, 3]
        bu_set = BudgetSet(budget_mat, budget_rhs_vec)
        exc_str = ".*must have 2 rows to match shape of attribute 'budget_rhs_vec' \\(provided.*1 rows\\)"
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.budget_membership_mat = [[1, 0, 1]]
        exc_str = ".*must have 2 entries to match shape of attribute 'budget_membership_mat' \\(provided.*1 entries\\)"
        with self.assertRaisesRegex(ValueError, exc_str):
            bu_set.budget_rhs_vec = [1]

    def test_error_on_neg_budget_rhs_vec_entry(self):
        """
        Test ValueError raised if budget RHS vec has entry
        with negative value entry.
        """
        budget_mat = [[1, 0, 1], [1, 1, 0]]
        neg_val_rhs_vec = [1, -1]
        exc_str = "Entry -1 of.*'budget_rhs_vec' is negative*"
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(budget_mat, neg_val_rhs_vec)
        buset = BudgetSet(budget_mat, [1, 1])
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_rhs_vec = neg_val_rhs_vec

    def test_error_on_non_bool_budget_mat_entry(self):
        """
        Test ValueError raised if budget membership mat has
        entry which is not a 0-1 value.
        """
        invalid_budget_mat = [[1, 0, 1], [1, 1, 0.1]]
        budget_rhs_vec = [1, 1]
        exc_str = 'Attempting.*entries.*not 0-1 values \\(example: 0.1\\).*'
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(invalid_budget_mat, budget_rhs_vec)
        buset = BudgetSet([[1, 0, 1], [1, 1, 0]], budget_rhs_vec)
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_membership_mat = invalid_budget_mat

    def test_error_on_budget_mat_all_zero_rows(self):
        """
        Test ValueError raised if budget membership mat
        has a row with all zeros.
        """
        invalid_row_mat = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]
        budget_rhs_vec = [1, 1, 2]
        exc_str = '.*all entries zero in rows at indexes: 0, 2.*'
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(invalid_row_mat, budget_rhs_vec)
        buset = BudgetSet([[1, 0, 1], [1, 1, 0], [1, 1, 1]], budget_rhs_vec)
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_membership_mat = invalid_row_mat

    def test_error_on_budget_mat_all_zero_columns(self):
        """
        Test ValueError raised if budget membership mat
        has a column with all zeros.
        """
        invalid_col_mat = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
        budget_rhs_vec = [1, 1, 2]
        exc_str = '.*all entries zero in columns at indexes: 0, 1.*'
        with self.assertRaisesRegex(ValueError, exc_str):
            BudgetSet(invalid_col_mat, budget_rhs_vec)
        buset = BudgetSet([[1, 0, 1], [1, 1, 0], [1, 1, 1]], budget_rhs_vec)
        with self.assertRaisesRegex(ValueError, exc_str):
            buset.budget_membership_mat = invalid_col_mat

    @unittest.skipUnless(SolverFactory('cbc').available(exception_flag=False), 'LP solver CBC not available')
    def test_budget_set_parameter_bounds_correct(self):
        """
        If LP solver is available, test parameter bounds method
        for factor model set is correct (check against
        results from an LP solver).
        """
        solver = SolverFactory('cbc')
        buset1 = BudgetSet(budget_membership_mat=[[1, 1], [0, 1]], rhs_vec=[2, 3], origin=None)
        buset2 = BudgetSet(budget_membership_mat=[[1, 0], [1, 1]], rhs_vec=[3, 2], origin=[1, 1])
        for buset in [buset1, buset2]:
            param_bounds = buset.parameter_bounds
            lp_param_bounds = eval_parameter_bounds(buset, solver)
            self.assertTrue(np.allclose(param_bounds, lp_param_bounds), msg=f'Parameter bounds not consistent with LP values for BudgetSet with parameterization:\nbudget_membership_mat={buset.budget_membership_mat},\nbudget_rhs_vec={buset.budget_rhs_vec},\norigin={buset.origin}.\n({param_bounds} does not match {lp_param_bounds})')

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
        budget_membership_mat = [[1 for i in range(len(m.uncertain_param_vars))]]
        rhs_vec = [0.1 * len(m.uncertain_param_vars) + sum((p.value for p in m.uncertain_param_vars.values()))]
        _set = BudgetSet(budget_membership_mat=budget_membership_mat, rhs_vec=rhs_vec)
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
        Case in which the BudgetSet is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        """
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        budget_membership_mat = [[1 for i in range(len(m.uncertain_param_vars))]]
        rhs_vec = [0.1 * len(m.uncertain_param_vars) + sum((p.value for p in m.uncertain_param_vars.values()))]
        _set = BudgetSet(budget_membership_mat=budget_membership_mat, rhs_vec=rhs_vec)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        vars_in_expr = []
        for con in m.uncertainty_set_contr.values():
            vars_in_expr.extend((v for v in m.uncertain_param_vars.values() if v in ComponentSet(identify_variables(expr=con.expr))))
        self.assertEqual(len(vars_in_expr), 0, msg='Uncertainty set constraint contains no Var objects, consists of a not potentially variable expression.')

    def test_budget_set_as_constraint(self):
        """
        The set_as_constraint method must return an indexed uncertainty_set_constr
        which has as many elements at their are dimensions in A.
        """
        m = ConcreteModel()
        m.p1 = Var(initialize=1)
        m.p2 = Var(initialize=1)
        m.uncertain_params = [m.p1, m.p2]
        budget_membership_mat = [[1 for i in range(len(m.uncertain_params))]]
        rhs_vec = [0.1 * len(m.uncertain_params) + sum((p.value for p in m.uncertain_params))]
        budget_set = BudgetSet(budget_membership_mat=budget_membership_mat, rhs_vec=rhs_vec)
        m.uncertainty_set_constr = budget_set.set_as_constraint(uncertain_params=m.uncertain_params)
        self.assertEqual(len(budget_set.coefficients_mat), len(m.uncertainty_set_constr.index_set()), msg="Number of budget set constraints should be equal to the number of rows in the 'coefficients_mat' attribute")

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        budget_membership_mat = [[1 for i in range(len(m.uncertain_params))]]
        rhs_vec = [0.1 * len(m.uncertain_params) + sum((p.value for p in m.uncertain_params))]
        budget_set = BudgetSet(budget_membership_mat=budget_membership_mat, rhs_vec=rhs_vec)
        self.assertTrue(budget_set.point_in_set([0, 0]), msg='Point is not in the BudgetSet.')

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)
        budget_membership_mat = [[1 for i in range(len(m.util.uncertain_param_vars))]]
        rhs_vec = [0.1 * len(m.util.uncertain_param_vars) + sum((value(p) for p in m.util.uncertain_param_vars.values()))]
        budget_set = BudgetSet(budget_membership_mat=budget_membership_mat, rhs_vec=rhs_vec)
        config = Block()
        config.uncertainty_set = budget_set
        BudgetSet.add_bounds_on_uncertain_parameters(model=m, config=config)
        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, 'Bounds not added correctly for BudgetSet')
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, 'Bounds not added correctly for BudgetSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, 'Bounds not added correctly for BudgetSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, 'Bounds not added correctly for BudgetSet')