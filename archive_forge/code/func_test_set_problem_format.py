import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.opt
def test_set_problem_format(self):
    opt = pyomo.opt.SolverFactory('stest1')
    opt._valid_problem_formats = []
    try:
        opt.set_problem_format('a')
    except ValueError:
        pass
    else:
        self.fail("Should not be able to set the problem format undless it's declared as valid.")
    opt._valid_problem_formats = ['a']
    self.assertEqual(opt.results_format(), None)
    opt.set_problem_format('a')
    self.assertEqual(opt.problem_format(), 'a')
    self.assertEqual(opt.results_format(), opt._default_results_format('a'))