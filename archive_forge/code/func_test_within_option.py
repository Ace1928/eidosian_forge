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
def test_within_option(self):
    """Test within option"""
    self.model.x = Var(within=Reals)
    self.construct()
    self.assertEqual(type(self.instance.x.domain), RealSet)
    self.assertEqual(self.instance.x.is_integer(), False)
    self.assertEqual(self.instance.x.is_binary(), False)
    self.assertEqual(self.instance.x.is_continuous(), True)