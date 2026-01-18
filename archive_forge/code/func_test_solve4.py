import json
import os
from os.path import join
from filecmp import cmp
import pyomo.common.unittest as unittest
import pyomo.common
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.core import ConcreteModel
from pyomo.opt import ResultsFormat, SolverResults, SolverFactory
def test_solve4(self):
    """Test ASL - test4.nl"""
    _log = TempfileManager.create_tempfile('.test_solve4.log')
    _out = TempfileManager.create_tempfile('.test_solve4.txt')
    results = self.asl.solve(join(currdir, 'test4.nl'), logfile=_log, suffixes=['.*'])
    results.write(filename=_out, times=False, format='json')
    _baseline = join(currdir, 'test4_asl.txt')
    with open(_out, 'r') as out, open(_baseline, 'r') as txt:
        self.assertStructuredAlmostEqual(json.load(txt), json.load(out), abstol=0.0001, allow_second_superset=True)