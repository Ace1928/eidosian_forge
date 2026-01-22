from pyomo.common import unittest
from pyomo.contrib.solver.solution import SolutionLoaderBase, PersistentSolutionLoader
class TestSolSolutionLoader(unittest.TestCase):

    def test_member_list(self):
        expected_list = ['load_vars', 'get_primals', 'get_duals', 'get_reduced_costs']
        method_list = [method for method in dir(SolutionLoaderBase) if method.startswith('_') is False]
        self.assertEqual(sorted(expected_list), sorted(method_list))