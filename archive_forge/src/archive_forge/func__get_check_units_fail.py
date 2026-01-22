import math
import pickle
from pyomo.common.errors import PyomoException
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.log import LoggingIntercept
from pyomo.util.check_units import assert_units_consistent, check_units_equivalent
from pyomo.core.expr import inequality
from pyomo.core.expr.numvalue import NumericConstant
import pyomo.core.expr as EXPR
from pyomo.core.base.units_container import (
from io import StringIO
def _get_check_units_fail(self, x, pyomo_units_container, expected_type=None, expected_error=InconsistentUnitsError):
    if expected_type is not None:
        self.assertEqual(expected_type, type(x))
    with self.assertRaises(expected_error):
        assert_units_consistent(x)
    with self.assertRaises(expected_error):
        pyomo_units_container.get_units(x)