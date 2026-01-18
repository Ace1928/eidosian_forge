from unittest import mock
import uuid
from neutron_lib.placement import utils as place_utils
from neutron_lib.tests import _base as base
def test_parse_rp_inventory_defaults(self):
    self.assertEqual({}, place_utils.parse_rp_inventory_defaults({}))
    self.assertRaises(ValueError, place_utils.parse_rp_inventory_defaults, {'allocation_ratio': '-1.0'})
    self.assertEqual({'allocation_ratio': 1.0}, place_utils.parse_rp_inventory_defaults({'allocation_ratio': '1.0'}))
    self.assertRaises(ValueError, place_utils.parse_rp_inventory_defaults, {'min_unit': '-1'})
    self.assertEqual({'min_unit': 1}, place_utils.parse_rp_inventory_defaults({'min_unit': '1'}))
    self.assertRaises(ValueError, place_utils.parse_rp_inventory_defaults, {'no such inventory parameter': 1})