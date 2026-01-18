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
def test_verify_vlan_range_invalid_range(self):
    for v in [(constants.MIN_VLAN_TAG, constants.MAX_VLAN_TAG + 2), (constants.MIN_VLAN_TAG + 4, constants.MIN_VLAN_TAG + 1)]:
        self.assertRaises(exceptions.NetworkVlanRangeError, utils.verify_vlan_range, v)