from openstack.network.v2 import default_security_group_rule
from openstack.tests.unit import base
class TestDefaultSecurityGroupRule(base.TestCase):

    def test_basic(self):
        sot = default_security_group_rule.DefaultSecurityGroupRule()
        self.assertEqual('default_security_group_rule', sot.resource_key)
        self.assertEqual('default_security_group_rules', sot.resources_key)
        self.assertEqual('/default-security-group-rules', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertDictEqual({'description': 'description', 'direction': 'direction', 'id': 'id', 'ether_type': 'ethertype', 'limit': 'limit', 'marker': 'marker', 'port_range_max': 'port_range_max', 'port_range_min': 'port_range_min', 'protocol': 'protocol', 'remote_group_id': 'remote_group_id', 'remote_address_group_id': 'remote_address_group_id', 'remote_ip_prefix': 'remote_ip_prefix', 'sort_dir': 'sort_dir', 'sort_key': 'sort_key', 'used_in_default_sg': 'used_in_default_sg', 'used_in_non_default_sg': 'used_in_non_default_sg'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = default_security_group_rule.DefaultSecurityGroupRule(**EXAMPLE)
        self.assertEqual(EXAMPLE['description'], sot.description)
        self.assertEqual(EXAMPLE['direction'], sot.direction)
        self.assertEqual(EXAMPLE['ethertype'], sot.ether_type)
        self.assertEqual(EXAMPLE['id'], sot.id)
        self.assertEqual(EXAMPLE['port_range_max'], sot.port_range_max)
        self.assertEqual(EXAMPLE['port_range_min'], sot.port_range_min)
        self.assertEqual(EXAMPLE['protocol'], sot.protocol)
        self.assertEqual(EXAMPLE['remote_group_id'], sot.remote_group_id)
        self.assertEqual(EXAMPLE['remote_address_group_id'], sot.remote_address_group_id)
        self.assertEqual(EXAMPLE['remote_ip_prefix'], sot.remote_ip_prefix)
        self.assertEqual(EXAMPLE['used_in_default_sg'], sot.used_in_default_sg)
        self.assertEqual(EXAMPLE['used_in_non_default_sg'], sot.used_in_non_default_sg)