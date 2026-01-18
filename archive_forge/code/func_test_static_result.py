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
def test_static_result(self):
    obj = MyObj.query(self.context)
    self.assertEqual(obj.bar, 'bar')
    result = obj.marco()
    self.assertEqual(result, 'polo')