from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
class TestConvertToBoolean(base.BaseTestCase):

    def test_convert_to_boolean_bool(self):
        self.assertIs(converters.convert_to_boolean(True), True)
        self.assertIs(converters.convert_to_boolean(False), False)

    def test_convert_to_boolean_int(self):
        self.assertIs(converters.convert_to_boolean(0), False)
        self.assertIs(converters.convert_to_boolean(1), True)
        self.assertRaises(n_exc.InvalidInput, converters.convert_to_boolean, 7)

    def test_convert_to_boolean_str(self):
        self.assertIs(converters.convert_to_boolean('True'), True)
        self.assertIs(converters.convert_to_boolean('true'), True)
        self.assertIs(converters.convert_to_boolean('False'), False)
        self.assertIs(converters.convert_to_boolean('false'), False)
        self.assertIs(converters.convert_to_boolean('0'), False)
        self.assertIs(converters.convert_to_boolean('1'), True)
        self.assertRaises(n_exc.InvalidInput, converters.convert_to_boolean, '7')

    def test_convert_to_boolean_if_not_none(self):
        self.assertIsNone(converters.convert_to_boolean_if_not_none(None))
        self.assertIs(converters.convert_to_boolean_if_not_none(1), True)