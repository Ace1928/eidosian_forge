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
@unittest.skipUnless(sympy_available, 'Sympy not available')
class TestLogicalToLinearBackmap(unittest.TestCase):

    def test_backmap_deprecated(self):
        m = ConcreteModel()
        m.s = RangeSet(3)
        m.Y = BooleanVar(m.s)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core.base', logging.WARNING):
            y1 = m.Y[1].get_associated_binary()
        self.assertIn('DEPRECATED: Relying on core.logical_to_linear to transform BooleanVars that do not appear in LogicalConstraints is deprecated. Please associate your own binaries if you have BooleanVars not used in logical expressions.', output.getvalue().replace('\n', ' '))
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core.base', logging.WARNING):
            y2 = m.Y[2].get_associated_binary()
        self.assertIn('DEPRECATED: Relying on core.logical_to_linear to transform BooleanVars that do not appear in LogicalConstraints is deprecated. Please associate your own binaries if you have BooleanVars not used in logical expressions.', output.getvalue().replace('\n', ' '))
        y1.value = 1
        y2.value = 0
        update_boolean_vars_from_binary(m)
        self.assertTrue(m.Y[1].value)
        self.assertFalse(m.Y[2].value)
        self.assertIsNone(m.Y[3].value)

    def test_can_associate_unused_boolean_after_transformation(self):
        m = ConcreteModel()
        m.Y = BooleanVar()
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.y = Var(domain=Binary)
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.core.base', logging.WARNING):
            m.Y.associate_binary_var(m.y)
            y = m.Y.get_associated_binary()
        self.assertIs(y, m.y)
        self.assertEqual(output.getvalue(), '')

    def test_cannot_reassociate_boolean_error(self):
        m = _generate_boolean_model(2)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.y = Var(domain=Binary)
        with self.assertRaisesRegex(RuntimeError, "Reassociating BooleanVar 'Y\\[1\\]' \\(currently associated with 'Y_asbinary\\[1\\]'\\) with 'y' is not allowed"):
            m.Y[1].associate_binary_var(m.y)

    def test_backmap(self):
        m = _generate_boolean_model(3)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.Y_asbinary[1].value = 1
        m.Y_asbinary[2].value = 0
        update_boolean_vars_from_binary(m)
        self.assertTrue(m.Y[1].value)
        self.assertFalse(m.Y[2].value)
        self.assertIsNone(m.Y[3].value)

    def test_backmap_hierarchical_model(self):
        m = _generate_boolean_model(3)
        m.b = Block()
        m.b.Y = BooleanVar()
        m.b.lc = LogicalConstraint(expr=m.Y[1].lor(m.b.Y))
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.Y_asbinary[1].value = 1
        m.Y_asbinary[2].value = 0
        m.b.Y.get_associated_binary().value = 1
        update_boolean_vars_from_binary(m)
        self.assertTrue(m.Y[1].value)
        self.assertFalse(m.Y[2].value)
        self.assertIsNone(m.Y[3].value)
        self.assertTrue(m.b.Y.value)

    def test_backmap_noninteger(self):
        m = _generate_boolean_model(2)
        TransformationFactory('core.logical_to_linear').apply_to(m)
        m.Y_asbinary[1].value = 0.9
        update_boolean_vars_from_binary(m, integer_tolerance=0.1)
        self.assertTrue(m.Y[1].value)
        with self.assertRaisesRegex(ValueError, 'Binary variable has non-\\{0,1\\} value'):
            update_boolean_vars_from_binary(m)