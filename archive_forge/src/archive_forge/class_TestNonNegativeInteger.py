import datetime
from unittest import mock
import warnings
import iso8601
import netaddr
import testtools
from oslo_versionedobjects import _utils
from oslo_versionedobjects import base as obj_base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import test
class TestNonNegativeInteger(TestField):

    def setUp(self):
        super(TestNonNegativeInteger, self).setUp()
        self.field = fields.NonNegativeIntegerField()
        self.coerce_good_values = [(1, 1), ('1', 1)]
        self.coerce_bad_values = ['-2', '4.2', 'foo', None]
        self.to_primitive_values = self.coerce_good_values[0:1]
        self.from_primitive_values = self.coerce_good_values[0:1]

    def test_get_schema(self):
        self.assertEqual({'type': ['integer'], 'readonly': False, 'minimum': 0}, self.field.get_schema())