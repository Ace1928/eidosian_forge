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
def test_validate_ipv6(self):
    testdata = '2001:0db8:0a0b:12f0:0000:0000:0000:0001'
    self.assertIsNone(validators.validate_ip_or_subnet_or_none(testdata))