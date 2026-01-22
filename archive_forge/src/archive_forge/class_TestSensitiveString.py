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
class TestSensitiveString(TestString):

    def setUp(self):
        super(TestSensitiveString, self).setUp()
        self.field = fields.SensitiveStringField()

    def test_stringify(self):
        payload = "{'admin_password':'mypassword'}"
        expected = "'{'admin_password':'***'}'"
        self.assertEqual(expected, self.field.stringify(payload))