from itertools import zip_longest
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
import pyomo.scripting.convert as convert
from pyomo.common.fileutils import this_file_dir, PYOMO_ROOT_DIR
from pyomo.common.tempfiles import TempfileManager
from pyomo.repn.tests.nl_diff import load_and_compare_nl_baseline
from pyomo.repn.tests.lp_diff import load_and_compare_lp_baseline
def test_nonconvex_nl(self):
    """Test examples/pyomo/piecewise/nonconvex.py"""
    self.run_convert2nl('nonconvex.py')
    _test, _base = (join(self.tmpdir, 'unknown.nl'), join(currdir, 'nonconvex.nl'))
    self.assertEqual(*load_and_compare_nl_baseline(_base, _test))