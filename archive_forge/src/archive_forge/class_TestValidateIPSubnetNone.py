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
class TestValidateIPSubnetNone(base.BaseTestCase):

    def test_validate_none(self):
        self.assertIsNone(validators.validate_ip_or_subnet_or_none(None))

    def test_validate_ipv4(self):
        testdata = '172.0.0.1'
        self.assertIsNone(validators.validate_ip_or_subnet_or_none(testdata))

    def test_validate_ipv4_subnet(self):
        testdata = '172.0.0.1/24'
        self.assertIsNone(validators.validate_ip_or_subnet_or_none(testdata))

    def test_validate_ipv6(self):
        testdata = '2001:0db8:0a0b:12f0:0000:0000:0000:0001'
        self.assertIsNone(validators.validate_ip_or_subnet_or_none(testdata))

    def test_validate_ipv6_subnet(self):
        testdata = '::1/128'
        self.assertIsNone(validators.validate_ip_or_subnet_or_none(testdata))

    def test_validate_ipv4_invalid(self):
        testdata = '300.0.0.1'
        self.assertEqual("'300.0.0.1' is neither a valid IP address, nor is it a valid IP subnet", validators.validate_ip_or_subnet_or_none(testdata))

    def test_validate_ipv4_subnet_invalid(self):
        testdata = '172.0.0.1/45'
        self.assertEqual("'172.0.0.1/45' is neither a valid IP address, nor is it a valid IP subnet", validators.validate_ip_or_subnet_or_none(testdata))

    def test_validate_ipv6_invalid(self):
        testdata = 'xxxx:0db8:0a0b:12f0:0000:0000:0000:0001'
        self.assertEqual("'xxxx:0db8:0a0b:12f0:0000:0000:0000:0001' is neither a valid IP address, nor is it a valid IP subnet", validators.validate_ip_or_subnet_or_none(testdata))

    def test_validate_ipv6_subnet_invalid(self):
        testdata = '::1/2048'
        self.assertEqual("'::1/2048' is neither a valid IP address, nor is it a valid IP subnet", validators.validate_ip_or_subnet_or_none(testdata))