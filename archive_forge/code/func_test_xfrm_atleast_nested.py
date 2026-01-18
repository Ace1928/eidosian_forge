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
def test_xfrm_atleast_nested(self):
    m = _generate_boolean_model(4)
    m.p = LogicalConstraint(expr=atleast(1, atleast(2, m.Y[1], m.Y[1].lor(m.Y[2]), m.Y[2]).lor(m.Y[3]), m.Y[4]))
    TransformationFactory('core.logical_to_linear').apply_to(m)
    Y_aug = m.logic_to_linear.augmented_vars
    self.assertEqual(len(Y_aug), 3)
    _constrs_contained_within(self, [(1, Y_aug[1].get_associated_binary() + m.Y[4].get_associated_binary(), None), (1, 1 - Y_aug[2].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (1, 1 - m.Y[3].get_associated_binary() + Y_aug[1].get_associated_binary(), None), (1, Y_aug[2].get_associated_binary() + m.Y[3].get_associated_binary() + 1 - Y_aug[1].get_associated_binary(), None), (1, 1 - m.Y[1].get_associated_binary() + Y_aug[3].get_associated_binary(), None), (1, 1 - m.Y[2].get_associated_binary() + Y_aug[3].get_associated_binary(), None), (1, m.Y[1].get_associated_binary() + m.Y[2].get_associated_binary() + 1 - Y_aug[3].get_associated_binary(), None), (None, 2 - 2 * (1 - Y_aug[2].get_associated_binary()) - (m.Y[1].get_associated_binary() + Y_aug[3].get_associated_binary() + m.Y[2].get_associated_binary()), 0), (None, m.Y[1].get_associated_binary() + Y_aug[3].get_associated_binary() + m.Y[2].get_associated_binary() - (1 + 2 * Y_aug[2].get_associated_binary()), 0)], m.logic_to_linear.transformed_constraints)