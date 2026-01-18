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
def test_obj_attr_is_set(self):
    obj = MyObj(foo=1)
    self.assertTrue(obj.obj_attr_is_set('foo'))
    self.assertFalse(obj.obj_attr_is_set('bar'))
    self.assertRaises(AttributeError, obj.obj_attr_is_set, 'bang')