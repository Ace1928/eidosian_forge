import os
from os.path import abspath, dirname
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.core.base import IntegerSet
from pyomo.core.expr.numeric_expr import (
from pyomo.core.staleflag import StaleFlagManager
from pyomo.environ import (
from pyomo.core.base.units_container import units, pint_available, UnitsError
def test_unfix_nonindexed(self):
    """Test unfix variables method"""
    self.model.B = RangeSet(4)
    self.model.x = Var()
    self.model.y = Var(self.model.B)
    self.instance = self.model.create_instance()
    self.instance.x.fixed = True
    self.instance.x.unfix()
    self.assertEqual(self.instance.x.fixed, False)