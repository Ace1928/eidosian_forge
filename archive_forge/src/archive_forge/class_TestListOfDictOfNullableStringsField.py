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
class TestListOfDictOfNullableStringsField(TestField):

    def setUp(self):
        super(TestListOfDictOfNullableStringsField, self).setUp()
        self.field = fields.ListOfDictOfNullableStringsField()
        self.coerce_good_values = [([{'f': 'b', 'f1': 'b1'}, {'f2': 'b2'}], [{'f': 'b', 'f1': 'b1'}, {'f2': 'b2'}]), ([{'f': 1}, {'f1': 'b1'}], [{'f': '1'}, {'f1': 'b1'}]), ([{'foo': None}], [{'foo': None}])]
        self.coerce_bad_values = [[{1: 'a'}], ['ham', 1], ['eggs']]
        self.to_primitive_values = [([{'f': 'b'}, {'f1': 'b1'}, {'f2': None}], [{'f': 'b'}, {'f1': 'b1'}, {'f2': None}])]
        self.from_primitive_values = [([{'f': 'b'}, {'f1': 'b1'}, {'f2': None}], [{'f': 'b'}, {'f1': 'b1'}, {'f2': None}])]

    def test_stringify(self):
        self.assertEqual("[{f=None,f1='b1'},{f2='b2'}]", self.field.stringify([{'f': None, 'f1': 'b1'}, {'f2': 'b2'}]))