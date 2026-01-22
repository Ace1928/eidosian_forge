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
class testCardinalityUncertaintySetClass(unittest.TestCase):
    """
    Cardinality uncertainty sets. Required inputs are origin, positive_deviation, gamma.
    Because Cardinality adds cassi vars to model, must pass model to set_as_constraint()
    """

    def test_normal_cardinality_construction_and_update(self):
        """
        Test CardinalitySet constructor and setter work normally
        when bounds are appropriate.
        """
        cset = CardinalitySet(origin=[0, 0], positive_deviation=[1, 3], gamma=2)
        np.testing.assert_allclose(cset.origin, [0, 0])
        np.testing.assert_allclose(cset.positive_deviation, [1, 3])
        np.testing.assert_allclose(cset.gamma, 2)
        self.assertEqual(cset.dim, 2)
        cset.origin = [1, 2]
        cset.positive_deviation = [3, 0]
        cset.gamma = 0.5
        np.testing.assert_allclose(cset.origin, [1, 2])
        np.testing.assert_allclose(cset.positive_deviation, [3, 0])
        np.testing.assert_allclose(cset.gamma, 0.5)

    def test_error_on_neg_positive_deviation(self):
        """
        Cardinality set positive deviation attribute should
        contain nonnegative numerical entries.

        Check ValueError raised if any negative entries provided.
        """
        origin = [0, 0]
        positive_deviation = [1, -2]
        gamma = 2
        exc_str = "Entry -2 of attribute 'positive_deviation' is negative value"
        with self.assertRaisesRegex(ValueError, exc_str):
            cset = CardinalitySet(origin, positive_deviation, gamma)
        cset = CardinalitySet(origin, [1, 1], gamma)
        with self.assertRaisesRegex(ValueError, exc_str):
            cset.positive_deviation = positive_deviation

    def test_error_on_invalid_gamma(self):
        """
        Cardinality set gamma attribute should be a float-like
        between 0 and the set dimension.

        Check ValueError raised if gamma attribute is set
        to an invalid value.
        """
        origin = [0, 0]
        positive_deviation = [1, 1]
        gamma = 3
        exc_str = ".*attribute 'gamma' must be a real number between 0 and dimension 2 \\(provided value 3\\)"
        with self.assertRaisesRegex(ValueError, exc_str):
            CardinalitySet(origin, positive_deviation, gamma)
        cset = CardinalitySet(origin, positive_deviation, gamma=2)
        with self.assertRaisesRegex(ValueError, exc_str):
            cset.gamma = gamma

    def test_error_on_cardinality_set_dim_change(self):
        """
        Dimension is considered immutable.
        Test ValueError raised when attempting to alter the
        set dimension (i.e. number of entries of `origin`).
        """
        cset = CardinalitySet(origin=[0, 0], positive_deviation=[1, 1], gamma=2)
        exc_str = 'Attempting to set.*dimension 2 to value of dimension 3'
        with self.assertRaisesRegex(ValueError, exc_str):
            cset.origin = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, exc_str):
            cset.positive_deviation = [1, 1, 1]

    @unittest.skipIf(not numpy_available, 'Numpy is not available.')
    def test_uncertainty_set_with_correct_params(self):
        """
        Case in which the UncertaintySet is constructed using the uncertain_param objects from the model to
        which the uncertainty set constraint is being added.
        """
        m = ConcreteModel()
        m.util = Block()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Var(range(len(m.uncertain_params)), initialize=0)
        center = list((p.value for p in m.uncertain_param_vars.values()))
        positive_deviation = list((0.3 for j in range(len(center))))
        gamma = np.ceil(len(m.uncertain_param_vars) / 2)
        _set = CardinalitySet(origin=center, positive_deviation=positive_deviation, gamma=gamma)
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
        Case in which the CardinalitySet is constructed using  uncertain_param objects which are Params instead of
        Vars. Leads to a constraint this is not potentially variable.
        """
        m = ConcreteModel()
        m.util = Block()
        m.p1 = Var(initialize=0)
        m.p2 = Var(initialize=0)
        m.uncertain_params = [m.p1, m.p2]
        m.uncertain_param_vars = Param(range(len(m.uncertain_params)), initialize=0, mutable=True)
        center = list((p.value for p in m.uncertain_param_vars.values()))
        positive_deviation = list((0.3 for j in range(len(center))))
        gamma = np.ceil(len(m.uncertain_param_vars) / 2)
        _set = CardinalitySet(origin=center, positive_deviation=positive_deviation, gamma=gamma)
        m.uncertainty_set_contr = _set.set_as_constraint(uncertain_params=m.uncertain_param_vars, model=m)
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
        center = list((p.value for p in m.uncertain_param_vars.values()))
        positive_deviation = list((0.3 for j in range(len(center))))
        gamma = np.ceil(len(m.uncertain_param_vars) / 2)
        _set = CardinalitySet(origin=center, positive_deviation=positive_deviation, gamma=gamma)
        self.assertTrue(_set.point_in_set([0, 0]), msg='Point is not in the CardinalitySet.')

    def test_add_bounds_on_uncertain_parameters(self):
        m = ConcreteModel()
        m.util = Block()
        m.util.uncertain_param_vars = Var([0, 1], initialize=0.5)
        center = list((p.value for p in m.util.uncertain_param_vars.values()))
        positive_deviation = list((0.3 for j in range(len(center))))
        gamma = np.ceil(len(center) / 2)
        cardinality_set = CardinalitySet(origin=center, positive_deviation=positive_deviation, gamma=gamma)
        config = Block()
        config.uncertainty_set = cardinality_set
        CardinalitySet.add_bounds_on_uncertain_parameters(model=m, config=config)
        self.assertNotEqual(m.util.uncertain_param_vars[0].lb, None, 'Bounds not added correctly for CardinalitySet')
        self.assertNotEqual(m.util.uncertain_param_vars[0].ub, None, 'Bounds not added correctly for CardinalitySet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].lb, None, 'Bounds not added correctly for CardinalitySet')
        self.assertNotEqual(m.util.uncertain_param_vars[1].ub, None, 'Bounds not added correctly for CardinalitySet')