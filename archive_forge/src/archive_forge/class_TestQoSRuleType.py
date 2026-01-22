from openstack.network.v2 import qos_rule_type
from openstack.tests.unit import base
class TestQoSRuleType(base.TestCase):

    def test_basic(self):
        sot = qos_rule_type.QoSRuleType()
        self.assertEqual('rule_type', sot.resource_key)
        self.assertEqual('rule_types', sot.resources_key)
        self.assertEqual('/qos/rule-types', sot.base_path)
        self.assertFalse(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertFalse(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual({'type': 'type', 'drivers': 'drivers', 'all_rules': 'all_rules', 'all_supported': 'all_supported', 'limit': 'limit', 'marker': 'marker'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = qos_rule_type.QoSRuleType(**EXAMPLE)
        self.assertEqual(EXAMPLE['type'], sot.type)
        self.assertEqual(EXAMPLE['drivers'], sot.drivers)