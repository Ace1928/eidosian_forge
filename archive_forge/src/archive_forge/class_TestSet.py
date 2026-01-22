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
class TestSet(TestField):

    def setUp(self):
        super(TestSet, self).setUp()
        self.field = fields.Field(fields.Set(FakeFieldType()))
        self.coerce_good_values = [(set(['foo', 'bar']), set(['*foo*', '*bar*']))]
        self.coerce_bad_values = [['foo'], {'foo': 'bar'}]
        self.to_primitive_values = [(set(['foo']), tuple(['!foo!']))]
        self.from_primitive_values = [(tuple(['!foo!']), set(['foo']))]

    def test_stringify(self):
        self.assertEqual('set([123])', self.field.stringify(set([123])))

    def test_get_schema(self):
        self.assertEqual({'type': ['array'], 'uniqueItems': True, 'items': {'type': ['foo'], 'readonly': False}, 'readonly': False}, self.field.get_schema())