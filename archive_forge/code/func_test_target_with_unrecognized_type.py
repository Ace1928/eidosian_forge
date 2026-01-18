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
def test_target_with_unrecognized_type(self):
    m = _generate_boolean_model(2)
    with self.assertRaisesRegex(ValueError, "invalid value for configuration 'targets':\\n\\tFailed casting 1\\n\\tto target_list\\n\\tError: Expected Component or list of Components.\\n\\tReceived <class 'int'>"):
        TransformationFactory('core.logical_to_linear').apply_to(m, targets=1)