import hashlib
from unittest import mock
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import uuidutils
from neutron_lib.api.definitions import portbindings_extended as pb_ext
from neutron_lib import constants
from neutron_lib import exceptions
from neutron_lib.plugins import utils
from neutron_lib.tests import _base as base
def test_parse_network_vlan_range_missing_network(self):
    self.assertRaises(exceptions.PhysicalNetworkNameError, utils.parse_network_vlan_range, ':1:4')