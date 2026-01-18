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
def test_obj_primitive_field_namespace_wrong(self):
    primitive = {'foo.name': 'TestObject', 'foo.namespace': 'wrong', 'foo.version': '1.0', 'foo.data': {}}
    self.assertRaises(exception.UnsupportedObjectError, self.test_class.obj_from_primitive, primitive)