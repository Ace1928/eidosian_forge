from openstack.baremetal_introspection.v1 import introspection_rule
from openstack.tests.unit import base
class TestIntrospectionRule(base.TestCase):

    def test_basic(self):
        sot = introspection_rule.IntrospectionRule()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('rules', sot.resources_key)
        self.assertEqual('/rules', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertFalse(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertEqual('POST', sot.create_method)

    def test_instantiate(self):
        sot = introspection_rule.IntrospectionRule(**FAKE)
        self.assertEqual(FAKE['conditions'], sot.conditions)
        self.assertEqual(FAKE['actions'], sot.actions)
        self.assertEqual(FAKE['description'], sot.description)
        self.assertEqual(FAKE['uuid'], sot.id)
        self.assertEqual(FAKE['scope'], sot.scope)