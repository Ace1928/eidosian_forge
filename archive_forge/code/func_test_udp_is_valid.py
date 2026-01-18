from unittest import mock
import netaddr
import testtools
from neutron_lib.api import converters
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib.tests import _base as base
from neutron_lib.tests import tools
def test_udp_is_valid(self):
    result = converters.convert_to_protocol(constants.PROTO_NAME_UDP)
    self.assertEqual(constants.PROTO_NAME_UDP, result)
    proto_num_str = str(constants.PROTO_NUM_UDP)
    result = converters.convert_to_protocol(proto_num_str)
    self.assertEqual(proto_num_str, result)