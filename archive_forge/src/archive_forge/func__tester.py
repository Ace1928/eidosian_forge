import subprocess
import sys
from math import nan, inf
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.environ import (
from pyomo.core.pyomoobject import PyomoObject
from pyomo.core.expr.numvalue import (
from pyomo.common.numeric_types import _native_boolean_types
def _tester(expr):
    rc = subprocess.run([sys.executable, '-c', cmd % expr], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    self.assertEqual((rc.returncode, rc.stdout), (0, 'False\nTrue\n'))