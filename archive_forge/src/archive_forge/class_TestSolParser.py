from pyomo.common import unittest
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
from pyomo.contrib.solver.sol_reader import parse_sol_file, SolFileData
class TestSolParser(unittest.TestCase):

    def setUp(self):
        TempfileManager.push()

    def tearDown(self):
        TempfileManager.pop(remove=True)

    def test_default_behavior(self):
        pass

    def test_custom_behavior(self):
        pass

    def test_infeasible1(self):
        pass

    def test_infeasible2(self):
        pass