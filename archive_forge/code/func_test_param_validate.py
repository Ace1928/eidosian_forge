import math
import os
import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.common.errors import PyomoException
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.core.base.param import _ParamData
from pyomo.core.base.set import _SetData
from pyomo.core.base.units_container import units, pint_available, UnitsError
from io import StringIO
def test_param_validate(self):
    """Test Param `validate` and `within` throw ValueError when not valid.

        The `within` argument will catch the ValueError, log extra information
        with of an "ERROR" message, and reraise the ValueError.

        1. Immutable Param (unindexed)
        2. Immutable Param (indexed)
        3. Immutable Param (arbitrary validation rule)
        4. Mutable Param (unindexed)
        5. Mutable Param (indexed)
        6. Mutable Param (arbitrary validation rule)
        """

    def validation_rule(model, value):
        """Arbitrary validation rule that always returns False."""
        return False
    with self.assertRaisesRegex(ValueError, 'Value not in parameter domain'):
        m = ConcreteModel()
        m.p1 = Param(initialize=-3, within=NonNegativeReals)
    with self.assertRaisesRegex(ValueError, 'Value not in parameter domain'):
        m = ConcreteModel()
        m.A = RangeSet(1, 2)
        m.p2 = Param(m.A, initialize=-3, within=NonNegativeReals)
    with self.assertRaisesRegex(ValueError, 'Invalid parameter value'):
        m = ConcreteModel()
        m.p5 = Param(initialize=1, validate=validation_rule)
    with self.assertRaisesRegex(ValueError, 'Value not in parameter domain'):
        m = ConcreteModel()
        m.p3 = Param(within=NonNegativeReals, mutable=True)
        m.p3 = -3
    with self.assertRaisesRegex(ValueError, 'Value not in parameter domain'):
        m = ConcreteModel()
        m.A = RangeSet(1, 2)
        m.p4 = Param(m.A, within=NonNegativeReals, mutable=True)
        m.p4[1] = -3
    with self.assertRaisesRegex(ValueError, 'Invalid parameter value'):
        m = ConcreteModel()
        m.p6 = Param(mutable=True, validate=validation_rule)
        m.p6 = 1
    a = AbstractModel()
    a.p = Param(within=NonNegativeReals)
    a.p = 1
    with self.assertRaisesRegex(ValueError, 'Value not in parameter domain'):
        a.p = -2
    with self.assertRaisesRegex(RuntimeError, 'Value not in parameter domain'):
        m = a.create_instance({None: {'p': {None: -1}}})
    m = a.create_instance()
    self.assertEqual(value(m.p), 1)