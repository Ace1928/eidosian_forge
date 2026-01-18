import os
from io import StringIO
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.core.base import NumericLabeler, SymbolMap
from pyomo.environ import (
from pyomo.gdp import Disjunction
from pyomo.network import Port, Arc
from pyomo.repn.plugins.gams_writer import (
def test_split_long_line_no_comment(self):
    pat = '1000 * 2000 * '
    line = pat * 5715 + 'x'
    self.assertEqual(split_long_line(line), pat * 5714 + '1000\n * 2000 * x')