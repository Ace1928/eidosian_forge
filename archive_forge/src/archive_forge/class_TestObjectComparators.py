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
class TestObjectComparators(test.TestCase):

    @base.VersionedObjectRegistry.register_if(False)
    class MyComparedObject(base.VersionedObject):
        fields = {'foo': fields.IntegerField(), 'bar': fields.IntegerField()}

    @base.VersionedObjectRegistry.register_if(False)
    class MyComparedObjectWithTZ(base.VersionedObject):
        fields = {'tzfield': fields.DateTimeField()}

    def test_compare_obj(self):
        mock_test = mock.Mock()
        mock_test.assertEqual = mock.Mock()
        my_obj = self.MyComparedObject(foo=1, bar=2)
        my_db_obj = {'foo': 1, 'bar': 2}
        fixture.compare_obj(mock_test, my_obj, my_db_obj)
        expected_calls = [(1, 1), (2, 2)]
        actual_calls = [c[0] for c in mock_test.assertEqual.call_args_list]
        for call in expected_calls:
            self.assertIn(call, actual_calls)

    def test_compare_obj_with_unset(self):
        mock_test = mock.Mock()
        mock_test.assertEqual = mock.Mock()
        my_obj = self.MyComparedObject()
        my_db_obj = {}
        fixture.compare_obj(mock_test, my_obj, my_db_obj)
        self.assertFalse(mock_test.assertEqual.called, 'assertEqual should not have been called, there is nothing to compare.')

    def test_compare_obj_with_unset_in_obj(self):
        mock_test = mock.Mock()
        mock_test.assertEqual = mock.Mock()
        my_obj = self.MyComparedObject(foo=1)
        my_db_obj = {'foo': 1, 'bar': 2}
        self.assertRaises(AssertionError, fixture.compare_obj, mock_test, my_obj, my_db_obj)

    def test_compare_obj_with_unset_in_db_dict(self):
        mock_test = mock.Mock()
        mock_test.assertEqual = mock.Mock()
        my_obj = self.MyComparedObject(foo=1, bar=2)
        my_db_obj = {'foo': 1}
        self.assertRaises(AssertionError, fixture.compare_obj, mock_test, my_obj, my_db_obj)

    def test_compare_obj_with_unset_in_obj_ignored(self):
        my_obj = self.MyComparedObject(foo=1)
        my_db_obj = {'foo': 1, 'bar': 2}
        ignore = ['bar']
        fixture.compare_obj(self, my_obj, my_db_obj, allow_missing=ignore)

    def test_compare_obj_with_unset_in_db_dict_ignored(self):
        my_obj = self.MyComparedObject(foo=1, bar=2)
        my_db_obj = {'foo': 1}
        ignore = ['bar']
        fixture.compare_obj(self, my_obj, my_db_obj, allow_missing=ignore)

    def test_compare_obj_with_allow_missing_unequal(self):
        mock_test = mock.Mock()
        mock_test.assertEqual = mock.Mock()
        my_obj = self.MyComparedObject(foo=1, bar=2)
        my_db_obj = {'foo': 1, 'bar': 1}
        ignore = ['bar']
        fixture.compare_obj(mock_test, my_obj, my_db_obj, allow_missing=ignore)
        expected_calls = [(1, 1), (1, 2)]
        actual_calls = [c[0] for c in mock_test.assertEqual.call_args_list]
        for call in expected_calls:
            self.assertIn(call, actual_calls)

    def test_compare_obj_with_subs(self):
        mock_test = mock.Mock()
        mock_test.assertEqual = mock.Mock()
        my_obj = self.MyComparedObject(foo=1, bar=2)
        my_db_obj = {'doo': 1, 'bar': 2}
        subs = {'foo': 'doo'}
        fixture.compare_obj(mock_test, my_obj, my_db_obj, subs=subs)
        expected_calls = [(1, 1), (2, 2)]
        actual_calls = [c[0] for c in mock_test.assertEqual.call_args_list]
        for call in expected_calls:
            self.assertIn(call, actual_calls)

    def test_compare_obj_with_allow_missing(self):
        mock_test = mock.Mock()
        mock_test.assertEqual = mock.Mock()
        my_obj = self.MyComparedObject(foo=1)
        my_db_obj = {'foo': 1, 'bar': 2}
        ignores = ['bar']
        fixture.compare_obj(mock_test, my_obj, my_db_obj, allow_missing=ignores)
        mock_test.assertEqual.assert_called_once_with(1, 1)

    def test_compare_obj_with_comparators(self):
        mock_test = mock.Mock()
        mock_test.assertEqual = mock.Mock()
        comparator = mock.Mock()
        comp_dict = {'foo': comparator}
        my_obj = self.MyComparedObject(foo=1, bar=2)
        my_db_obj = {'foo': 1, 'bar': 2}
        fixture.compare_obj(mock_test, my_obj, my_db_obj, comparators=comp_dict)
        comparator.assert_called_once_with(1, 1)
        mock_test.assertEqual.assert_called_once_with(2, 2)

    def test_compare_obj_with_dt(self):
        mock_test = mock.Mock()
        mock_test.assertEqual = mock.Mock()
        dt = datetime.datetime(1955, 11, 5, tzinfo=iso8601.iso8601.UTC)
        replaced_dt = dt.replace(tzinfo=None)
        my_obj = self.MyComparedObjectWithTZ(tzfield=dt)
        my_db_obj = {'tzfield': replaced_dt}
        fixture.compare_obj(mock_test, my_obj, my_db_obj)
        mock_test.assertEqual.assert_called_once_with(replaced_dt, replaced_dt)