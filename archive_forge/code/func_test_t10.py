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
def test_t10(self):
    M = self._setup()

    def f(model, i):
        return complements(M.y + M.x3, M.x1 + 2 * M.x2 == i)
    M.cc = Complementarity([0, 1, 2], rule=f)
    M.cc[1].deactivate()
    self._test('t10', M)