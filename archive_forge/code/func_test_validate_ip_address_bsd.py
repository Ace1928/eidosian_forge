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
def test_validate_ip_address_bsd(self):
    ip_addr = '1' * 59
    with mock.patch('netaddr.IPAddress') as ip_address_cls:
        msg = validators.validate_ip_address(ip_addr)
    ip_address_cls.assert_called_once_with(ip_addr, flags=netaddr.core.ZEROFILL)
    self.assertEqual("'%s' is not a valid IP address" % ip_addr, msg)