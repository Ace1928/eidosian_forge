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
def test_validate_nameservers(self):
    ns_pools = [['1.1.1.2', '1.1.1.2'], ['www.hostname.com', 'www.hostname.com'], ['1000.0.0.1'], ['www.hostname.com'], ['www.great.marathons.to.travel'], ['valid'], ['77.hostname.com'], ['1' * 59], ['www.internal.hostname.com'], None]
    for ns in ns_pools:
        msg = validators.validate_nameservers(ns, None)
        self.assertIsNotNone(msg)
    ns_pools = [['100.0.0.2'], ['1.1.1.1', '1.1.1.2']]
    for ns in ns_pools:
        msg = validators.validate_nameservers(ns, None)
        self.assertIsNone(msg)