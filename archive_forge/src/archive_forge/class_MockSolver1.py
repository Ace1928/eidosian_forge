import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
class MockSolver1(pyomo.opt.OptSolver):

    def __init__(self, **kwds):
        kwds['type'] = 'stest_type'
        kwds['doc'] = 'MockSolver1 Documentation'
        pyomo.opt.OptSolver.__init__(self, **kwds)