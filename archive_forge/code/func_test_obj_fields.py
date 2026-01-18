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
def test_obj_fields(self):

    class TestObj(base.VersionedObject):
        fields = {'foo': fields.Field(fields.Integer())}
        obj_extra_fields = ['bar']

        @property
        def bar(self):
            return 'this is bar'
    obj = TestObj()
    self.assertEqual(['foo', 'bar'], obj.obj_fields)