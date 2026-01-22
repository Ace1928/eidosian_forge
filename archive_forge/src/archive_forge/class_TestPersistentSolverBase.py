import os
from pyomo.common import unittest
from pyomo.common.config import ConfigDict
from pyomo.contrib.solver import base
class TestPersistentSolverBase(unittest.TestCase):

    def test_abstract_member_list(self):
        expected_list = ['remove_parameters', 'version', 'update_variables', 'remove_variables', 'add_constraints', '_get_primals', 'set_instance', 'set_objective', 'update_parameters', 'remove_block', 'add_block', 'available', 'add_parameters', 'remove_constraints', 'add_variables', 'solve']
        member_list = list(base.PersistentSolverBase.__abstractmethods__)
        self.assertEqual(sorted(expected_list), sorted(member_list))

    def test_class_method_list(self):
        expected_list = ['Availability', 'CONFIG', '_get_duals', '_get_primals', '_get_reduced_costs', '_load_vars', 'add_block', 'add_constraints', 'add_parameters', 'add_variables', 'available', 'is_persistent', 'remove_block', 'remove_constraints', 'remove_parameters', 'remove_variables', 'set_instance', 'set_objective', 'solve', 'update_parameters', 'update_variables', 'version']
        method_list = [method for method in dir(base.PersistentSolverBase) if (method.startswith('__') or method.startswith('_abc')) is False]
        self.assertEqual(sorted(expected_list), sorted(method_list))

    @unittest.mock.patch.multiple(base.PersistentSolverBase, __abstractmethods__=set())
    def test_init(self):
        self.instance = base.PersistentSolverBase()
        self.assertTrue(self.instance.is_persistent())
        self.assertEqual(self.instance.set_instance(None), None)
        self.assertEqual(self.instance.add_variables(None), None)
        self.assertEqual(self.instance.add_parameters(None), None)
        self.assertEqual(self.instance.add_constraints(None), None)
        self.assertEqual(self.instance.add_block(None), None)
        self.assertEqual(self.instance.remove_variables(None), None)
        self.assertEqual(self.instance.remove_parameters(None), None)
        self.assertEqual(self.instance.remove_constraints(None), None)
        self.assertEqual(self.instance.remove_block(None), None)
        self.assertEqual(self.instance.set_objective(None), None)
        self.assertEqual(self.instance.update_variables(None), None)
        self.assertEqual(self.instance.update_parameters(), None)
        with self.assertRaises(NotImplementedError):
            self.instance._get_primals()
        with self.assertRaises(NotImplementedError):
            self.instance._get_duals()
        with self.assertRaises(NotImplementedError):
            self.instance._get_reduced_costs()

    @unittest.mock.patch.multiple(base.PersistentSolverBase, __abstractmethods__=set())
    def test_context_manager(self):
        with base.PersistentSolverBase() as self.instance:
            self.assertTrue(self.instance.is_persistent())
            self.assertEqual(self.instance.set_instance(None), None)
            self.assertEqual(self.instance.add_variables(None), None)
            self.assertEqual(self.instance.add_parameters(None), None)
            self.assertEqual(self.instance.add_constraints(None), None)
            self.assertEqual(self.instance.add_block(None), None)
            self.assertEqual(self.instance.remove_variables(None), None)
            self.assertEqual(self.instance.remove_parameters(None), None)
            self.assertEqual(self.instance.remove_constraints(None), None)
            self.assertEqual(self.instance.remove_block(None), None)
            self.assertEqual(self.instance.set_objective(None), None)
            self.assertEqual(self.instance.update_variables(None), None)
            self.assertEqual(self.instance.update_parameters(), None)