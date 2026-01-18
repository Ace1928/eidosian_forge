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
def test_validate_ip_address(self):
    ip_addr = '1.1.1.1'
    msg = validators.validate_ip_address(ip_addr)
    self.assertIsNone(msg)
    ip_addr = '1111.1.1.1'
    msg = validators.validate_ip_address(ip_addr)
    self.assertEqual("'%s' is not a valid IP address" % ip_addr, msg)
    ip_addr = '1' * 59
    msg = validators.validate_ip_address(ip_addr)
    self.assertEqual("'%s' is not a valid IP address" % ip_addr, msg)
    ip_addr = '1.1.1.1 has whitespace'
    msg = validators.validate_ip_address(ip_addr)
    self.assertEqual("'%s' is not a valid IP address" % ip_addr, msg)
    ip_addr = '111.1.1.1\twhitespace'
    msg = validators.validate_ip_address(ip_addr)
    self.assertEqual("'%s' is not a valid IP address" % ip_addr, msg)
    ip_addr = '111.1.1.1\nwhitespace'
    msg = validators.validate_ip_address(ip_addr)
    self.assertEqual("'%s' is not a valid IP address" % ip_addr, msg)
    for ws in string.whitespace:
        ip_addr = '%s111.1.1.1' % ws
        msg = validators.validate_ip_address(ip_addr)
        self.assertEqual("'%s' is not a valid IP address" % ip_addr, msg)
    for ws in string.whitespace:
        ip_addr = '111.1.1.1%s' % ws
        msg = validators.validate_ip_address(ip_addr)
        self.assertEqual("'%s' is not a valid IP address" % ip_addr, msg)