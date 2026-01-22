from itertools import zip_longest
import re
import sys
import os
from os.path import join
from filecmp import cmp
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.opt import ProblemFormat, ConverterError, convert_problem
from pyomo.common import Executable
class MockArg4(MockArg):

    def write(self, filename='', format=None, solver_capability=None, io_options={}):
        OUTPUT = open(filename, 'w')
        INPUT = open(join(currdir, 'test4.nl'))
        for line in INPUT:
            OUTPUT.write(line)
        OUTPUT.close()
        INPUT.close()
        return (filename, None)