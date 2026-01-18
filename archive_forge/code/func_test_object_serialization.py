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
def test_object_serialization(self):
    ser = base.VersionedObjectSerializer()
    obj = MyObj()
    primitive = ser.serialize_entity(self.context, obj)
    self.assertIn('versioned_object.name', primitive)
    obj2 = ser.deserialize_entity(self.context, primitive)
    self.assertIsInstance(obj2, MyObj)
    self.assertEqual(self.context, obj2._context)