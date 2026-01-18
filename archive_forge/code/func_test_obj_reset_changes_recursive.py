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
def test_obj_reset_changes_recursive(self):
    obj = MyObj(rel_object=MyOwnedObject(baz=123), rel_objects=[MyOwnedObject(baz=456)])
    self.assertEqual(set(['rel_object', 'rel_objects']), obj.obj_what_changed())
    obj.obj_reset_changes()
    self.assertEqual(set(['rel_object']), obj.obj_what_changed())
    self.assertEqual(set(['baz']), obj.rel_object.obj_what_changed())
    self.assertEqual(set(['baz']), obj.rel_objects[0].obj_what_changed())
    obj.obj_reset_changes(recursive=True, fields=['foo'])
    self.assertEqual(set(['rel_object']), obj.obj_what_changed())
    self.assertEqual(set(['baz']), obj.rel_object.obj_what_changed())
    self.assertEqual(set(['baz']), obj.rel_objects[0].obj_what_changed())
    obj.obj_reset_changes(recursive=True)
    self.assertEqual(set([]), obj.rel_object.obj_what_changed())
    self.assertEqual(set([]), obj.obj_what_changed())