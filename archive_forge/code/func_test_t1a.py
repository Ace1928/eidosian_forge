import os
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import (
from pyomo.gdp import Disjunct, Disjunction
from pyomo.mpec import Complementarity, complements, ComplementarityList
from pyomo.opt import ProblemFormat
from pyomo.repn.plugins.nl_writer import FileDeterminism
from pyomo.repn.tests.nl_diff import load_and_compare_nl_baseline
def test_t1a(self):
    M = self._setup()
    M.c = Constraint(expr=M.y + M.x3 >= M.x2)
    M.cc = Complementarity(expr=complements(M.y + M.x1 >= 0, M.x1 + 2 * M.x2 + 3 * M.x3 >= 1))
    self._test('t1a', M)