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
class TestUtilityMethods(test.TestCase):

    def test_flat(self):

        @base.VersionedObjectRegistry.register
        class TestObject(base.VersionedObject):
            VERSION = '1.23'
            fields = {}
        tree = base.obj_tree_get_versions('TestObject')
        self.assertEqual({'TestObject': '1.23'}, tree)

    def test_parent_child(self):

        @base.VersionedObjectRegistry.register
        class TestChild(base.VersionedObject):
            VERSION = '2.34'

        @base.VersionedObjectRegistry.register
        class TestObject(base.VersionedObject):
            VERSION = '1.23'
            fields = {'child': fields.ObjectField('TestChild')}
        tree = base.obj_tree_get_versions('TestObject')
        self.assertEqual({'TestObject': '1.23', 'TestChild': '2.34'}, tree)

    def test_complex(self):

        @base.VersionedObjectRegistry.register
        class TestChild(base.VersionedObject):
            VERSION = '2.34'

        @base.VersionedObjectRegistry.register
        class TestChildTwo(base.VersionedObject):
            VERSION = '4.56'
            fields = {'sibling': fields.ObjectField('TestChild')}

        @base.VersionedObjectRegistry.register
        class TestObject(base.VersionedObject):
            VERSION = '1.23'
            fields = {'child': fields.ObjectField('TestChild'), 'childtwo': fields.ListOfObjectsField('TestChildTwo')}
        tree = base.obj_tree_get_versions('TestObject')
        self.assertEqual({'TestObject': '1.23', 'TestChild': '2.34', 'TestChildTwo': '4.56'}, tree)

    def test_complex_loopy(self):

        @base.VersionedObjectRegistry.register
        class TestChild(base.VersionedObject):
            VERSION = '2.34'
            fields = {'sibling': fields.ObjectField('TestChildTwo')}

        @base.VersionedObjectRegistry.register
        class TestChildTwo(base.VersionedObject):
            VERSION = '4.56'
            fields = {'sibling': fields.ObjectField('TestChild'), 'parents': fields.ListOfObjectsField('TestObject')}

        @base.VersionedObjectRegistry.register
        class TestObject(base.VersionedObject):
            VERSION = '1.23'
            fields = {'child': fields.ObjectField('TestChild'), 'childtwo': fields.ListOfObjectsField('TestChildTwo')}
        tree = base.obj_tree_get_versions('TestObject')
        self.assertEqual({'TestObject': '1.23', 'TestChild': '2.34', 'TestChildTwo': '4.56'}, tree)

    def test_missing_referenced(self):
        """Ensure a missing child object is highlighted."""

        @base.VersionedObjectRegistry.register
        class TestObjectFoo(base.VersionedObject):
            VERSION = '1.23'
            fields = {'child': fields.ObjectField('TestChildBar')}
        exc = self.assertRaises(exception.UnregisteredSubobject, base.obj_tree_get_versions, 'TestObjectFoo')
        self.assertIn('TestChildBar is referenced by TestObjectFoo', exc.format_message())