from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_parse_rp_pp_with_direction(self):
    self.assertEqual({}, place_utils.parse_rp_pp_with_direction([], 'default_host'))
    self.assertEqual({'host0': {'egress': None, 'ingress': None}}, place_utils.parse_rp_pp_with_direction(['host0'], 'default_host'))
    self.assertEqual({'host0': {'egress': None, 'ingress': None}}, place_utils.parse_rp_pp_with_direction(['host0::'], 'default_host'))
    self.assertEqual({'default_host': {'egress': None, 'ingress': None}}, place_utils.parse_rp_pp_with_direction(['::'], 'default_host'))
    self.assertEqual({'host0': {'egress': 1, 'ingress': None}}, place_utils.parse_rp_pp_with_direction(['host0:1:'], 'default_host'))
    self.assertEqual({'host0': {'egress': None, 'ingress': 1}}, place_utils.parse_rp_pp_with_direction(['host0::1'], 'default_host'))
    self.assertEqual({'host0': {'egress': 1, 'ingress': 1}}, place_utils.parse_rp_pp_with_direction(['host0:1:1'], 'default_host'))
    self.assertEqual({'default_host': {'egress': 0, 'ingress': 0}}, place_utils.parse_rp_pp_with_direction([':0:0'], 'default_host'))
    self.assertEqual({'host0': {'egress': 1, 'ingress': 1}, 'host1': {'egress': 10, 'ingress': 10}}, place_utils.parse_rp_pp_with_direction(['host0:1:1', 'host1:10:10'], 'default_host'))
    self.assertRaises(ValueError, place_utils.parse_rp_pp_with_direction, ['default_host::', '::'], 'default_host')
    self.assertRaises(ValueError, place_utils.parse_rp_pp_with_direction, ['host0::', 'host0::'], 'default_host')
    self.assertRaises(ValueError, place_utils.parse_rp_pp_with_direction, ['host0:not a number:not a number'], 'default_host')