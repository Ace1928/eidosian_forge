import glob
import sys
from os.path import basename, dirname, abspath, join
import subprocess
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, scipy_available
import platform
def testmethod(self):
    if basename(example) == 'piecewise_nd_functions.py':
        if not numpy_available or not scipy_available or (not testing_solvers['ipopt', 'nl']) or (not testing_solvers['glpk', 'lp']):
            self.skipTest('Numpy or Scipy or Ipopt or Glpk is not available')
    elif 'mosek' in example:
        if not testing_solvers['ipopt', 'nl'] or not testing_solvers['mosek_direct', 'python']:
            self.skipTest('Ipopt or Mosek is not available')
    result = subprocess.run([sys.executable, example], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    self.assertEqual(result.returncode, 0, msg=result.stdout)