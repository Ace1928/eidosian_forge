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
def test_verify_tunnel_range_invalid_range(self):
    for r in [[1, 0], [0, -1], [2, 1]]:
        self.assertRaises(exceptions.NetworkTunnelRangeError, utils.verify_tunnel_range, r, constants.TYPE_FLAT)