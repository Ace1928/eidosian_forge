import os
from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver import base
class TestLegacySolverWrapper(unittest.TestCase):

    def test_class_method_list(self):
        expected_list = ['available', 'license_is_valid', 'solve']
        method_list = [method for method in dir(base.LegacySolverWrapper) if method.startswith('_') is False]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    def test_context_manager(self):
        with base.LegacySolverWrapper() as instance:
            with self.assertRaises(AttributeError):
                instance.available()

    def test_map_config(self):
        self.config = ConfigDict(implicit=True)
        self.config.declare('solver_options', ConfigDict(implicit=True, description='Options to pass to the solver.'))
        instance = base.LegacySolverWrapper()
        instance.config = self.config
        instance._map_config(True, False, False, 20, True, False, None, None, None, False, None, None)
        self.assertTrue(instance.config.tee)
        self.assertFalse(instance.config.load_solutions)
        self.assertEqual(instance.config.time_limit, 20)
        with self.assertRaises(AttributeError):
            print(instance.config.report_timing)
        with self.assertRaises(AttributeError):
            print(instance.config.keepfiles)
        with self.assertRaises(NotImplementedError):
            instance._map_config(False, False, False, 20, False, False, None, None, '/path/to/bogus/file', False, None, None)
        with self.assertRaises(NotImplementedError):
            instance._map_config(False, False, False, 20, False, False, None, '/path/to/bogus/file', None, False, None, None)
        with self.assertRaises(NotImplementedError):
            instance._map_config(False, False, False, 20, False, False, '/path/to/bogus/file', None, None, False, None, None)
        instance._map_config(False, False, False, 20, False, False, None, None, None, True, None, None)
        self.assertEqual(instance.config.working_dir, os.getcwd())
        with self.assertRaises(AttributeError):
            print(instance.config.keepfiles)

    def test_map_results(self):
        pass

    def test_solution_handler(self):
        pass