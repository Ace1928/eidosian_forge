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
def test_obj_read_only(self):
    obj = MyObj(context=self.context, foo=123, bar='abc')
    obj.readonly = 1
    self.assertRaises(exception.ReadOnlyFieldError, setattr, obj, 'readonly', 2)