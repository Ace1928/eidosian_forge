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
def test_test_relationships_none_changed(self):
    dep_tree = {}
    self._add_dependency(MyObject, MyObject2, dep_tree)
    with mock.patch.object(self.ovc, 'get_dependency_tree') as mock_gdt:
        mock_gdt.return_value = dep_tree
        actual_exp, actual_act = self.ovc.test_relationships(dep_tree)
    expected_expected = expected_actual = {}
    self.assertEqual(expected_expected, actual_exp, "There are no objects changed, so the 'expected' return value should contain no objects.")
    self.assertEqual(expected_actual, actual_act, "There are no objects changed, so the 'actual' return value should contain no objects.")