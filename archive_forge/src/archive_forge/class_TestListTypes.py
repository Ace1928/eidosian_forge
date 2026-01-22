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
class TestListTypes(test.TestCase):

    def test_regular_list(self):
        fields.List(fields.Integer).coerce(None, None, [1, 2])

    def test_non_iterable(self):
        self.assertRaises(ValueError, fields.List(fields.Integer).coerce, None, None, 2)

    def test_string_iterable(self):
        self.assertRaises(ValueError, fields.List(fields.Integer).coerce, None, None, 'hello')

    def test_mapping_iterable(self):
        self.assertRaises(ValueError, fields.List(fields.Integer).coerce, None, None, {'a': 1, 'b': 2})

    def test_iter_class(self):
        fields.List(fields.Integer).coerce(None, None, FakeCounter())