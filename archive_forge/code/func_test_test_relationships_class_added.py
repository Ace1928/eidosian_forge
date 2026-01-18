import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_test_relationships_class_added(self):
    exp_tree = {}
    actual_tree = {}
    self._add_dependency(MyObject, MyObject2, exp_tree)
    self._add_dependency(MyObject, MyObject2, actual_tree)
    self._add_dependency(MyObject2, MyExtraObject, actual_tree)
    with mock.patch.object(self.ovc, 'get_dependency_tree') as mock_gdt:
        mock_gdt.return_value = actual_tree
        actual_exp, actual_act = self.ovc.test_relationships(exp_tree)
    expected_expected = {'MyObject2': None}
    expected_actual = {'MyObject2': {'MyExtraObject': '1.0'}}
    self.assertEqual(expected_expected, actual_exp, 'The expected relationship tree is not being built from changes correctly.')
    self.assertEqual(expected_actual, actual_act, 'The actual relationship tree is not being built from changes correctly.')