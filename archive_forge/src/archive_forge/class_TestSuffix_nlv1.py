import os
import pyomo.common.unittest as unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import ProblemFormat
from pyomo.environ import (
from ..nl_diff import load_and_compare_nl_baseline
class TestSuffix_nlv1(SuffixTester, unittest.TestCase):
    nl_version = 'nl_v1'