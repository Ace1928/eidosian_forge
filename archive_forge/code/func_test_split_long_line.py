import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_split_long_line(self):
    pat = 'var1 + log(var2 / 9) - '
    line = pat * 10000 + 'x'
    self.assertEqual(split_long_line(line), pat * 3478 + 'var1 +\n log(var2 / 9) - ' + pat * 3477 + 'var1 +\n log(var2 / 9) - ' + pat * 3043 + 'x')