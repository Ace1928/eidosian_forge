import glob
import sys
from os.path import basename, dirname, abspath, join
import subprocess
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import numpy_available, scipy_available
import platform
class TestKernelExamples(unittest.TestCase):
    pass