from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_parse_rp_bandwidths(self):
    self.assertEqual({}, place_utils.parse_rp_bandwidths([]))
    self.assertEqual({'eth0': {'egress': None, 'ingress': None}}, place_utils.parse_rp_bandwidths(['eth0']))
    self.assertEqual({'eth0': {'egress': None, 'ingress': None}}, place_utils.parse_rp_bandwidths(['eth0::']))
    self.assertRaises(ValueError, place_utils.parse_rp_bandwidths, ['eth0::', 'eth0::'])
    self.assertRaises(ValueError, place_utils.parse_rp_bandwidths, ['eth0:not a number:not a number'])
    self.assertEqual({'eth0': {'egress': 1, 'ingress': None}}, place_utils.parse_rp_bandwidths(['eth0:1:']))
    self.assertEqual({'eth0': {'egress': None, 'ingress': 1}}, place_utils.parse_rp_bandwidths(['eth0::1']))
    self.assertEqual({'eth0': {'egress': 1, 'ingress': 1}}, place_utils.parse_rp_bandwidths(['eth0:1:1']))
    self.assertEqual({'eth0': {'egress': 1, 'ingress': 1}, 'eth1': {'egress': 10, 'ingress': 10}}, place_utils.parse_rp_bandwidths(['eth0:1:1', 'eth1:10:10']))