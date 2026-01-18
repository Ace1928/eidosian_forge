import pyomo.common.unittest as unittest
from pyomo.common.errors import MouseTrap, DeveloperError
from pyomo.common.log import LoggingIntercept
import logging
from pyomo.core.expr.sympy_tools import sympy_available
from pyomo.core.plugins.transform.logical_to_linear import (
from pyomo.environ import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.repn import generate_standard_repn
from io import StringIO
def test_xfrm_special_atoms_nonroot(self):
    m = ConcreteModel()
    m.s = RangeSet(3)
    m.Y = BooleanVar(m.s)
    m.p = LogicalConstraint(expr=m.Y[1].implies(atleast(2, m.Y[1], m.Y[2], m.Y[3])))
    TransformationFactory('core.logical_to_linear').apply_to(m)
    Y_aug = m.logic_to_linear.augmented_vars
    self.assertEqual(len(Y_aug), 1)
    self.assertEqual(Y_aug[1].domain, BooleanSet)
    _constrs_contained_within(self, [(None, sum(m.Y[:].get_associated_binary()) - (1 + 2 * Y_aug[1].get_associated_binary()), 0), (1, 1 - m.Y[1].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (None, 2 - 2 * (1 - Y_aug[1].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0)], m.logic_to_linear.transformed_constraints)
    m = ConcreteModel()
    m.s = RangeSet(3)
    m.Y = BooleanVar(m.s)
    m.p = LogicalConstraint(expr=m.Y[1].implies(atmost(2, m.Y[1], m.Y[2], m.Y[3])))
    TransformationFactory('core.logical_to_linear').apply_to(m)
    Y_aug = m.logic_to_linear.augmented_vars
    self.assertEqual(len(Y_aug), 1)
    self.assertEqual(Y_aug[1].domain, BooleanSet)
    _constrs_contained_within(self, [(None, sum(m.Y[:].get_associated_binary()) - (1 - Y_aug[1].get_associated_binary() + 2), 0), (1, 1 - m.Y[1].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (None, 3 - 3 * Y_aug[1].get_associated_binary() - sum(m.Y[:].get_associated_binary()), 0)], m.logic_to_linear.transformed_constraints)
    m = ConcreteModel()
    m.s = RangeSet(3)
    m.Y = BooleanVar(m.s)
    m.p = LogicalConstraint(expr=m.Y[1].implies(exactly(2, m.Y[1], m.Y[2], m.Y[3])))
    TransformationFactory('core.logical_to_linear').apply_to(m)
    Y_aug = m.logic_to_linear.augmented_vars
    self.assertEqual(len(Y_aug), 3)
    self.assertEqual(Y_aug[1].domain, BooleanSet)
    _constrs_contained_within(self, [(1, 1 - m.Y[1].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (None, sum(m.Y[:].get_associated_binary()) - (1 - Y_aug[1].get_associated_binary() + 2), 0), (None, 2 - 2 * (1 - Y_aug[1].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0), (1, sum(Y_aug[:].get_associated_binary()), None), (None, sum(m.Y[:].get_associated_binary()) - (1 + 2 * (1 - Y_aug[2].get_associated_binary())), 0), (None, 3 - 3 * (1 - Y_aug[3].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0)], m.logic_to_linear.transformed_constraints)
    m = ConcreteModel()
    m.s = RangeSet(3)
    m.Y = BooleanVar(m.s)
    m.x = Var(bounds=(1, 3))
    m.p = LogicalConstraint(expr=m.Y[1].implies(exactly(m.x, m.Y[1], m.Y[2], m.Y[3])))
    TransformationFactory('core.logical_to_linear').apply_to(m)
    Y_aug = m.logic_to_linear.augmented_vars
    self.assertEqual(len(Y_aug), 3)
    self.assertEqual(Y_aug[1].domain, BooleanSet)
    _constrs_contained_within(self, [(1, 1 - m.Y[1].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (None, sum(m.Y[:].get_associated_binary()) - (m.x + 2 * (1 - Y_aug[1].get_associated_binary())), 0), (None, m.x - 3 * (1 - Y_aug[1].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0), (1, sum(Y_aug[:].get_associated_binary()), None), (None, sum(m.Y[:].get_associated_binary()) - (m.x - 1 + 3 * (1 - Y_aug[2].get_associated_binary())), 0), (None, m.x + 1 - 4 * (1 - Y_aug[3].get_associated_binary()) - sum(m.Y[:].get_associated_binary()), 0)], m.logic_to_linear.transformed_constraints)