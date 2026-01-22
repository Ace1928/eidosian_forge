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
class TestBoolean(TestField):

    def setUp(self):
        super(TestField, self).setUp()
        self.field = fields.BooleanField()
        self.coerce_good_values = [(True, True), (False, False), (1, True), ('foo', True), (0, False), ('', False)]
        self.coerce_bad_values = []
        self.to_primitive_values = self.coerce_good_values[0:2]
        self.from_primitive_values = self.coerce_good_values[0:2]

    def test_fieldtype_get_schema(self):
        self.assertEqual({'type': ['boolean']}, self.field._type.get_schema())

    def test_get_schema(self):
        self.assertEqual({'type': ['boolean'], 'readonly': False}, self.field.get_schema())