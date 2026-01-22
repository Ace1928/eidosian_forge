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
class testEllipsoidalUncertaintySetClass(unittest.TestCase):
    """
    Unit tests for the EllipsoidalSet
    """

    def test_normal_construction_and_update(self):
        """
        Test EllipsoidalSet constructor and setter
        work normally when arguments are appropriate.
        """
        center = [0, 0]
        shape_matrix = [[1, 0], [0, 2]]
        scale = 2
        eset = EllipsoidalSet(center, shape_matrix, scale)
        np.testing.assert_allclose(center, eset.center, err_msg='EllipsoidalSet center not as expected')
        np.testing.assert_allclose(shape_matrix, eset.shape_matrix, err_msg='EllipsoidalSet shape matrix not as expected')
        np.testing.assert_allclose(scale, eset.scale, err_msg='EllipsoidalSet scale not as expected')
        new_center = [-1, -3]
        new_shape_matrix = [[2, 1], [1, 3]]
        new_scale = 1
        eset.center = new_center
        eset.shape_matrix = new_shape_matrix
        eset.scale = new_scale
        np.testing.assert_allclose(new_center, eset.center, err_msg='EllipsoidalSet center update not as expected')
        np.testing.assert_allclose(new_shape_matrix, eset.shape_matrix, err_msg='EllipsoidalSet shape matrix update not as expected')
        np.testing.assert_allclose(new_scale, eset.scale, err_msg='EllipsoidalSet scale update not as expected')

    def test_error_on_ellipsoidal_dim_change(self):
        """
        EllipsoidalSet dimension is considered immutable.
        Test ValueError raised when center size is not equal
        to set dimension.
        """
        invalid_center = [0, 0]
        shape_matrix = [[1, 0], [0, 1]]
        scale = 2
        eset = EllipsoidalSet([0, 0], shape_matrix, scale)
        exc_str = 'Attempting to set.*dimension 2 to value of dimension 3'
        with self.assertRaisesRegex(ValueError, exc_str):
            eset.center = [0, 0, 0]

    def test_error_on_neg_scale(self):
        """
        Test ValueError raised if scale attribute set to negative
        value.
        """
        center = [0, 0]
        shape_matrix = [[1, 0], [0, 2]]
        neg_scale = -1
        exc_str = '.*must be a non-negative real \\(provided.*-1\\)'
        with self.assertRaisesRegex(ValueError, exc_str):
            EllipsoidalSet(center, shape_matrix, neg_scale)
        eset = EllipsoidalSet(center, shape_matrix, scale=2)
        with self.assertRaisesRegex(ValueError, exc_str):
            eset.scale = neg_scale

    def test_error_on_shape_matrix_with_wrong_size(self):
        """
        Test error in event EllipsoidalSet shape matrix
        is not in accordance with set dimension.
        """
        center = [0, 0]
        invalid_shape_matrix = [[1, 0]]
        scale = 1
        exc_str = '.*must be a square matrix of size 2.*\\(provided.*shape \\(1, 2\\)\\)'
        with self.assertRaisesRegex(ValueError, exc_str):
            EllipsoidalSet(center, invalid_shape_matrix, scale)
        eset = EllipsoidalSet(center, [[1, 0], [0, 1]], scale)
        with self.assertRaisesRegex(ValueError, exc_str):
            eset.shape_matrix = invalid_shape_matrix

    def test_error_on_invalid_shape_matrix(self):
        """
        Test exceptional cases of invalid square shape matrix
        arguments
        """
        center = [0, 0]
        scale = 3
        with self.assertRaisesRegex(ValueError, 'Shape matrix must be symmetric', msg='Asymmetric shape matrix test failed'):
            EllipsoidalSet(center, [[1, 1], [0, 1]], scale)
        with self.assertRaises(np.linalg.LinAlgError, msg='Singular shape matrix test failed'):
            EllipsoidalSet(center, [[0, 0], [0, 0]], scale)
        with self.assertRaisesRegex(ValueError, 'Non positive-definite.*', msg='Indefinite shape matrix test failed'):
            EllipsoidalSet(center, [[1, 0], [0, -2]], scale)
        eset = EllipsoidalSet(center, [[1, 0], [0, 2]], scale)
        with self.assertRaisesRegex(ValueError, 'Shape matrix must be symmetric', msg='Asymmetric shape matrix test failed'):
            eset.shape_matrix = [[1, 1], [0, 1]]
        with self.assertRaises(np.linalg.LinAlgError, msg='Singular shape matrix test failed'):
            eset.shape_matrix = [[0, 0], [0, 0]]
        with self.assertRaisesRegex(ValueError, 'Non positive-definite.*', msg='Indefinite shape matrix test failed'):
            eset.shape_matrix = [[1, 0], [0, -2]]

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
        cov = [[1, 0], [0, 1]]
        s = 1
        _set = EllipsoidalSet(center=[0, 0], shape_matrix=cov, scale=s)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        uncertain_params_in_expr = list((v for v in m.uncertain_param_vars.values() if v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))))
        self.assertEqual([id(u) for u in uncertain_params_in_expr], [id(u) for u in m.uncertain_param_vars.values()], msg='Uncertain param Var objects used to construct uncertainty set constraint must be the same uncertain param Var objects in the original model.')

    def test_uncertainty_set_with_incorrect_params(self):
        """
        Case in which the EllipsoidalSet is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        """
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        cov = [[1, 0], [0, 1]]
        s = 1
        _set = EllipsoidalSet(center=[0, 0], shape_matrix=cov, scale=s)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars)
        variables_in_constr = list((v for v in m.uncertain_params if v in ComponentSet(identify_variables(expr=m.uncertainty_set_contr[1].expr))))
        self.assertEqual(len(variables_in_constr), 0, msg='Uncertainty set constraint contains no Var objects, consists of a not potentially variable expression.')

    def test_point_in_set(self):
        m = ConcreteModel()
        m.p1 = Param(initialize=0, mutable=True)
        m.p2 = Param(initialize=0, mutable=True)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        cov = [[1, 0], [0, 1]]
        s = 1
        _set = EllipsoidalSet(center=[0, 0], shape_matrix=cov, scale=s)
        self.assertTrue(_set.point_in_set([0, 0]), msg='Point is not in the EllipsoidalSet.')

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)
        cov = [[1, 0], [0, 1]]
        s = 1
        _set = EllipsoidalSet(center=[0, 0], shape_matrix=cov, scale=s)
        config = Block()
        config.uncertainty_set = _set
        EllipsoidalSet.add_bounds_on_uncertain_parameters(model=m, config=config)
        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, 'Bounds not added correctly for EllipsoidalSet')
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, 'Bounds not added correctly for EllipsoidalSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, 'Bounds not added correctly for EllipsoidalSet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, 'Bounds not added correctly for EllipsoidalSet')

    def test_ellipsoidal_set_bounds(self):
        """Check `EllipsoidalSet` parameter bounds method correct."""
        cov = [[2, 1], [1, 2]]
        scales = [0.5, 2]
        mean = [1, 1]
        for scale in scales:
            ell = EllipsoidalSet(center=mean, shape_matrix=cov, scale=scale)
            bounds = ell.parameter_bounds
            actual_bounds = list()
            for idx, val in enumerate(mean):
                diff = (cov[idx][idx] * scale) ** 0.5
                actual_bounds.append((val - diff, val + diff))
            self.assertTrue(np.allclose(np.array(bounds), np.array(actual_bounds)), msg=f'EllipsoidalSet bounds {bounds} do not match their actual values {actual_bounds} (for scale {scale} and shape matrix {cov}). Check the `parameter_bounds` method for the EllipsoidalSet.')