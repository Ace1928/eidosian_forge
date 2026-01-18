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
def test_set_defaults_not_overwrite(self):
    obj = MyObj(deleted=True)
    obj.obj_set_defaults()
    self.assertEqual(1, obj.foo)
    self.assertTrue(obj.deleted)