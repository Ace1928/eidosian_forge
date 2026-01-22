import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
class TestRemoteObject(_RemoteTest, _TestObject):

    @mock.patch('oslo_versionedobjects.base.obj_tree_get_versions')
    def test_major_version_mismatch(self, mock_otgv):
        mock_otgv.return_value = {'MyObj': '2.0'}
        self.assertRaises(exception.IncompatibleObjectVersion, MyObj2.query, self.context)

    @mock.patch('oslo_versionedobjects.base.obj_tree_get_versions')
    def test_minor_version_greater(self, mock_otgv):
        mock_otgv.return_value = {'MyObj': '1.7'}
        self.assertRaises(exception.IncompatibleObjectVersion, MyObj2.query, self.context)

    @mock.patch('oslo_versionedobjects.base.obj_tree_get_versions')
    def test_minor_version_less(self, mock_otgv):
        mock_otgv.return_value = {'MyObj': '1.2'}
        obj = MyObj2.query(self.context)
        self.assertEqual(obj.bar, 'bar')

    @mock.patch('oslo_versionedobjects.base.obj_tree_get_versions')
    def test_compat(self, mock_otgv):
        mock_otgv.return_value = {'MyObj': '1.1'}
        obj = MyObj2.query(self.context)
        self.assertEqual('oldbar', obj.bar)

    @mock.patch('oslo_versionedobjects.base.obj_tree_get_versions')
    def test_revision_ignored(self, mock_otgv):
        mock_otgv.return_value = {'MyObj': '1.1.456'}
        obj = MyObj2.query(self.context)
        self.assertEqual('bar', obj.bar)

    def test_class_action_falls_back_compat(self):
        with mock.patch.object(base.VersionedObject, 'indirection_api') as ma:
            ma.object_class_action_versions.side_effect = NotImplementedError
            MyObj.query(self.context)
            ma.object_class_action.assert_called_once_with(self.context, 'MyObj', 'query', MyObj.VERSION, (), {})