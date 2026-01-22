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
class testFactorModelUncertaintySetClass(unittest.TestCase):
    """
    FactorModelSet uncertainty sets. Required inputs are psi_matrix, number_of_factors, origin and beta.
    """

    def test_normal_factor_model_construction_and_update(self):
        """
        Test FactorModelSet constructor and setter work normally
        when attribute values are appropriate.
        """
        fset = FactorModelSet(origin=[0, 0, 1], number_of_factors=2, psi_mat=[[1, 2], [0, 1], [1, 0]], beta=0.1)
        np.testing.assert_allclose(fset.origin, [0, 0, 1])
        np.testing.assert_allclose(fset.psi_mat, [[1, 2], [0, 1], [1, 0]])
        np.testing.assert_allclose(fset.number_of_factors, 2)
        np.testing.assert_allclose(fset.beta, 0.1)
        self.assertEqual(fset.dim, 3)
        fset.origin = [1, 1, 0]
        fset.psi_mat = [[1, 0], [0, 1], [1, 1]]
        fset.beta = 0.5
        np.testing.assert_allclose(fset.origin, [1, 1, 0])
        np.testing.assert_allclose(fset.psi_mat, [[1, 0], [0, 1], [1, 1]])
        np.testing.assert_allclose(fset.beta, 0.5)

    def test_error_on_factor_model_set_dim_change(self):
        """
        Test ValueError raised when attempting to change FactorModelSet
        dimension (by changing number of entries in origin
        or number of rows of psi_mat).
        """
        origin = [0, 0, 0]
        number_of_factors = 2
        psi_mat = [[1, 0], [0, 1], [1, 1]]
        beta = 0.5
        fset = FactorModelSet(origin, number_of_factors, psi_mat, beta)
        exc_str = 'should be of shape \\(3, 2\\) to match.*dimensions \\(provided shape \\(2, 2\\)\\)'
        with self.assertRaisesRegex(ValueError, exc_str):
            fset.psi_mat = [[1, 0], [1, 2]]
        exc_str = 'Attempting.*factor model set of dimension 3 to value of dimension 2'
        with self.assertRaisesRegex(ValueError, exc_str):
            fset.origin = [1, 3]

    def test_error_on_invalid_number_of_factors(self):
        """
        Test ValueError raised if number of factors
        is negative int, or AttributeError
        if attempting to update (should be immutable).
        """
        exc_str = ".*'number_of_factors' must be a positive int \\(provided value -1\\)"
        with self.assertRaisesRegex(ValueError, exc_str):
            FactorModelSet(origin=[0], number_of_factors=-1, psi_mat=[[1, 1]], beta=0.1)
        fset = FactorModelSet(origin=[0], number_of_factors=2, psi_mat=[[1, 1]], beta=0.1)
        exc_str = ".*'number_of_factors' is immutable"
        with self.assertRaisesRegex(AttributeError, exc_str):
            fset.number_of_factors = 3

    def test_error_on_invalid_beta(self):
        """
        Test ValueError raised if beta is invalid (exceeds 1 or
        is negative)
        """
        origin = [0, 0, 0]
        number_of_factors = 2
        psi_mat = [[1, 0], [0, 1], [1, 1]]
        neg_beta = -0.5
        big_beta = 1.5
        neg_exc_str = '.*must be a real number between 0 and 1.*\\(provided value -0.5\\)'
        big_exc_str = '.*must be a real number between 0 and 1.*\\(provided value 1.5\\)'
        with self.assertRaisesRegex(ValueError, neg_exc_str):
            FactorModelSet(origin, number_of_factors, psi_mat, neg_beta)
        with self.assertRaisesRegex(ValueError, big_exc_str):
            FactorModelSet(origin, number_of_factors, psi_mat, big_beta)
        fset = FactorModelSet(origin, number_of_factors, psi_mat, 1)
        with self.assertRaisesRegex(ValueError, neg_exc_str):
            fset.beta = neg_beta
        with self.assertRaisesRegex(ValueError, big_exc_str):
            fset.beta = big_beta

    @unittest.skipUnless(SolverFactory('cbc').available(exception_flag=False), 'LP solver CBC not available')
    def test_factor_model_parameter_bounds_correct(self):
        """
        If LP solver is available, test parameter bounds method
        for factor model set is correct (check against
        results from an LP solver).
        """
        solver = SolverFactory('cbc')
        fset1 = FactorModelSet(origin=[0, 0], number_of_factors=3, psi_mat=[[1, -1, 1], [1, 0.1, 1]], beta=1 / 6)
        fset2 = FactorModelSet(origin=[0], number_of_factors=3, psi_mat=[[1, 6, 8]], beta=1 / 2)
        fset3 = FactorModelSet(origin=[1], number_of_factors=2, psi_mat=[[1, 2]], beta=1 / 4)
        fset4 = FactorModelSet(origin=[1], number_of_factors=3, psi_mat=[[-1, -6, -8]], beta=1 / 2)
        for fset in [fset1, fset2, fset3, fset4]:
            param_bounds = fset.parameter_bounds
            lp_param_bounds = eval_parameter_bounds(fset, solver)
            self.assertTrue(np.allclose(param_bounds, lp_param_bounds), msg=f'Parameter bounds not consistent with LP values for FactorModelSet with parameterization:\nF={fset.number_of_factors},\nbeta={fset.beta},\npsi_mat={fset.psi_mat},\norigin={fset.origin}.')

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_uncertainty_set_with_correct_params(self):
        """
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        """
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.util = Block()
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        F = 1
        psi_mat = np.zeros(shape=(len(m.uncertain_params), F))
        for i in range(len(psi_mat)):
            random_row_entries = list(np.random.uniform(low=0, high=0.2, size=F))
            for j in range(len(psi_mat[i])):
                psi_mat[i][j] = random_row_entries[j]
        _set = FactorModelSet(origin=[0, 0], psi_mat=psi_mat, number_of_factors=F, beta=1)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars, model=m)
        uncertain_params_in_expr = []
        for con in m.uncertainty_set_contr.values():
            for v in m.uncertain_param_vars.values():
                if v in ComponentSet(identify_variables(expr=con.expr)):
                    if id(v) not in list((id(u) for u in uncertain_params_in_expr)):
                        uncertain_params_in_expr.append(v)
        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()], msg='Uncertain param Var objects used to construct uncertainty set constraint must be the same uncertain param Var objects in the original model.')

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_uncertainty_set_with_incorrect_params(self):
        """
        Case in which the set is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        """
        m = ConcreteModel()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.util = Block()
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        F = 1
        psi_mat = np.zeros(shape=(len(m.uncertain_params), F))
        for i in range(len(psi_mat)):
            random_row_entries = list(np.random.uniform(low=0, high=0.2, size=F))
            for j in range(len(psi_mat[i])):
                psi_mat[i][j] = random_row_entries[j]
        _set = FactorModelSet(origin=[0, 0], psi_mat=psi_mat, number_of_factors=F, beta=1)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars, model=m)
        vars_in_expr = []
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
        F = 1
        psi_mat = np.zeros(shape=(len(m.uncertain_params), F))
        for i in range(len(psi_mat)):
            random_row_entries = list(np.random.uniform(low=0, high=0.2, size=F))
            for j in range(len(psi_mat[i])):
                psi_mat[i][j] = random_row_entries[j]
        _set = FactorModelSet(origin=[0, 0], psi_mat=psi_mat, number_of_factors=F, beta=1)
        self.assertTrue(_set.point_in_set([0, 0]), msg='Point is not in the FactorModelSet.')

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0)
        F = 1
        psi_mat = np.zeros(shape=(len(list(m.util.uncertain_param_vars.values())), F))
        for i in range(len(psi_mat)):
            random_row_entries = list(np.random.uniform(low=0, high=0.2, size=F))
            for j in range(len(psi_mat[i])):
                psi_mat[i][j] = random_row_entries[j]
        _set = FactorModelSet(origin=[0, 0], psi_mat=psi_mat, number_of_factors=F, beta=1)
        config = Block()
        config.uncertainty_set = _set
        FactorModelSet.add_bounds_on_uncertain_parameters(model=m, config=config)
        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, 'Bounds not added correctly for FactorModelSet')
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, 'Bounds not added correctly for FactorModelSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, 'Bounds not added correctly for FactorModelSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, 'Bounds not added correctly for FactorModelSet')