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
def test_obj_primitive_field_namespace(self):
    primitive = {'foo.name': 'TestObject', 'foo.namespace': 'tests', 'foo.version': '1.0', 'foo.data': {}}
    with mock.patch.object(self.test_class, 'obj_class_from_name'):
        self.test_class.obj_from_primitive(primitive)