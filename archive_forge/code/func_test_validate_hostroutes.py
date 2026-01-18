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
def test_validate_hostroutes(self):
    hostroute_pools = [[{'destination': '100.0.0.0/24'}], [{'nexthop': '10.0.2.20'}], [{'nexthop': '10.0.2.20', 'forza': 'juve', 'destination': '100.0.0.0/8'}], [{'nexthop': '1110.0.2.20', 'destination': '100.0.0.0/8'}], [{'nexthop': '10.0.2.20', 'destination': '100.0.0.0'}], [{'nexthop': '10.0.2.20', 'destination': '100.0.0.0/8'}, {'nexthop': '10.0.2.20', 'destination': '100.0.0.0/8'}], [None], None]
    for host_routes in hostroute_pools:
        msg = validators.validate_hostroutes(host_routes, None)
        self.assertIsNotNone(msg)
    hostroute_pools = [[{'destination': '100.0.0.0/24', 'nexthop': '10.0.2.20'}], [{'nexthop': '10.0.2.20', 'destination': '100.0.0.0/8'}, {'nexthop': '10.0.2.20', 'destination': '101.0.0.0/8'}]]
    for host_routes in hostroute_pools:
        msg = validators.validate_hostroutes(host_routes, None)
        self.assertIsNone(msg)