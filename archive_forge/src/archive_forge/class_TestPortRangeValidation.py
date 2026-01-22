import string
from unittest import mock
import netaddr
from neutron_lib._i18n import _
from neutron_lib.api import converters
from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib import fixture
from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
class TestPortRangeValidation(base.BaseTestCase):

    def test_valid_port_specific_range(self):
        result = validators.validate_port_range_or_none('4:5', [1, 65535])
        self.assertIsNone(result)

    def test_invalid_port_specific_range(self):
        result = validators.validate_port_range_or_none('4:500000', [1, 65535])
        self.assertEqual(u'Invalid port: 500000', result)

    def test_invalid_port_for_specific_range(self):
        result = validators.validate_port_range_or_none('0:10', [1, 65535])
        self.assertEqual(u'Invalid port: 0, the port must be in the range [1, 65535]', result)

    def test_valid_port(self):
        result = validators.validate_port_range_or_none('80')
        self.assertIsNone(result)

    def test_valid_port_integer(self):
        result = validators.validate_port_range_or_none(80)
        self.assertIsNone(result)

    def test_valid_range(self):
        result = validators.validate_port_range_or_none('9:1111')
        self.assertIsNone(result)

    def test_port_too_high(self):
        result = validators.validate_port_range_or_none('99999')
        self.assertEqual(u'Invalid port: 99999', result)

    def test_port_too_low(self):
        result = validators.validate_port_range_or_none('-1')
        self.assertEqual(u'Invalid port: -1', result)

    def test_range_too_high(self):
        result = validators.validate_port_range_or_none('80:99999')
        self.assertEqual(u'Invalid port: 99999', result)

    def test_range_too_low(self):
        result = validators.validate_port_range_or_none('-1:8888')
        self.assertEqual(u'Invalid port: -1', result)

    def test_range_wrong_way(self):
        result = validators.validate_port_range_or_none('1111:9')
        self.assertEqual(u'First port in a port range must be lower than the second port', result)

    def test_range_invalid(self):
        result = validators.validate_port_range_or_none('DEAD:BEEF')
        self.assertEqual(u'Invalid port: DEAD', result)

    def test_range_bad_input(self):
        result = validators.validate_port_range_or_none(['a', 'b', 'c'])
        self.assertEqual(u"Invalid port: ['a', 'b', 'c']", result)

    def test_range_colon(self):
        result = validators.validate_port_range_or_none(':')
        self.assertEqual(u'Port range must be two integers separated by a colon', result)

    def test_too_many_colons(self):
        result = validators.validate_port_range_or_none('80:888:8888')
        self.assertEqual(u'Port range must be two integers separated by a colon', result)