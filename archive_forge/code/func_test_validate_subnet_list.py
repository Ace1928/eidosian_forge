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
def test_validate_subnet_list(self):
    msg = validators.validate_subnet_list('abc')
    self.assertEqual(u"'abc' is not a list", msg)
    msg = validators.validate_subnet_list(['10.1.0.0/24', '10.2.0.0/24', '10.1.0.0/24'])
    self.assertEqual(u"Duplicate items in the list: '10.1.0.0/24'", msg)
    msg = validators.validate_subnet_list(['10.1.0.0/24', '10.2.0.0'])
    self.assertEqual(u"'10.2.0.0' isn't a recognized IP subnet cidr, '10.2.0.0/32' is recommended", msg)