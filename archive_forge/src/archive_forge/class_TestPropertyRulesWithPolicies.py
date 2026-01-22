from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
class TestPropertyRulesWithPolicies(base.IsolatedUnitTest):

    def setUp(self):
        super(TestPropertyRulesWithPolicies, self).setUp()
        self.set_property_protections(use_policies=True)
        self.policy = policy.Enforcer(suppress_deprecation_warnings=True)
        self.rules_checker = property_utils.PropertyRules(self.policy)

    def test_check_property_rules_create_permitted_specific_policy(self):
        self.assertTrue(self.rules_checker.check_property_rules('spl_creator_policy', 'create', create_context(self.policy, ['spl_role'])))

    def test_check_property_rules_create_unpermitted_policy(self):
        self.assertFalse(self.rules_checker.check_property_rules('spl_creator_policy', 'create', create_context(self.policy, ['fake-role'])))

    def test_check_property_rules_read_permitted_specific_policy(self):
        self.assertTrue(self.rules_checker.check_property_rules('spl_creator_policy', 'read', create_context(self.policy, ['spl_role'])))

    def test_check_property_rules_read_unpermitted_policy(self):
        self.assertFalse(self.rules_checker.check_property_rules('spl_creator_policy', 'read', create_context(self.policy, ['fake-role'])))

    def test_check_property_rules_update_permitted_specific_policy(self):
        self.assertTrue(self.rules_checker.check_property_rules('spl_creator_policy', 'update', create_context(self.policy, ['admin'])))

    def test_check_property_rules_update_unpermitted_policy(self):
        self.assertFalse(self.rules_checker.check_property_rules('spl_creator_policy', 'update', create_context(self.policy, ['fake-role'])))

    def test_check_property_rules_delete_permitted_specific_policy(self):
        self.assertTrue(self.rules_checker.check_property_rules('spl_creator_policy', 'delete', create_context(self.policy, ['admin'])))

    def test_check_property_rules_delete_unpermitted_policy(self):
        self.assertFalse(self.rules_checker.check_property_rules('spl_creator_policy', 'delete', create_context(self.policy, ['fake-role'])))

    def test_property_protection_with_malformed_rule(self):
        malformed_rules = {'^[0-9)': {'create': ['fake-policy'], 'read': ['fake-policy'], 'update': ['fake-policy'], 'delete': ['fake-policy']}}
        self.set_property_protection_rules(malformed_rules)
        self.assertRaises(exception.InvalidPropertyProtectionConfiguration, property_utils.PropertyRules)

    def test_property_protection_with_multiple_policies(self):
        malformed_rules = {'^x_.*': {'create': ['fake-policy, another_pol'], 'read': ['fake-policy'], 'update': ['fake-policy'], 'delete': ['fake-policy']}}
        self.set_property_protection_rules(malformed_rules)
        self.assertRaises(exception.InvalidPropertyProtectionConfiguration, property_utils.PropertyRules)

    def test_check_property_rules_create_all_permitted(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertTrue(self.rules_checker.check_property_rules('x_all_permitted', 'create', create_context(self.policy, [''])))

    def test_check_property_rules_read_all_permitted(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertTrue(self.rules_checker.check_property_rules('x_all_permitted', 'read', create_context(self.policy, [''])))

    def test_check_property_rules_update_all_permitted(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertTrue(self.rules_checker.check_property_rules('x_all_permitted', 'update', create_context(self.policy, [''])))

    def test_check_property_rules_delete_all_permitted(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertTrue(self.rules_checker.check_property_rules('x_all_permitted', 'delete', create_context(self.policy, [''])))

    def test_check_property_rules_create_none_permitted(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertFalse(self.rules_checker.check_property_rules('x_none_permitted', 'create', create_context(self.policy, [''])))

    def test_check_property_rules_read_none_permitted(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertFalse(self.rules_checker.check_property_rules('x_none_permitted', 'read', create_context(self.policy, [''])))

    def test_check_property_rules_update_none_permitted(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertFalse(self.rules_checker.check_property_rules('x_none_permitted', 'update', create_context(self.policy, [''])))

    def test_check_property_rules_delete_none_permitted(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertFalse(self.rules_checker.check_property_rules('x_none_permitted', 'delete', create_context(self.policy, [''])))

    def test_check_property_rules_read_none(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertTrue(self.rules_checker.check_property_rules('x_none_read', 'create', create_context(self.policy, ['admin', 'member'])))
        self.assertFalse(self.rules_checker.check_property_rules('x_none_read', 'read', create_context(self.policy, [''])))
        self.assertFalse(self.rules_checker.check_property_rules('x_none_read', 'update', create_context(self.policy, [''])))
        self.assertFalse(self.rules_checker.check_property_rules('x_none_read', 'delete', create_context(self.policy, [''])))

    def test_check_property_rules_update_none(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertTrue(self.rules_checker.check_property_rules('x_none_update', 'create', create_context(self.policy, ['admin', 'member'])))
        self.assertTrue(self.rules_checker.check_property_rules('x_none_update', 'read', create_context(self.policy, ['admin', 'member'])))
        self.assertFalse(self.rules_checker.check_property_rules('x_none_update', 'update', create_context(self.policy, [''])))
        self.assertTrue(self.rules_checker.check_property_rules('x_none_update', 'delete', create_context(self.policy, ['admin', 'member'])))

    def test_check_property_rules_delete_none(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertTrue(self.rules_checker.check_property_rules('x_none_delete', 'create', create_context(self.policy, ['admin', 'member'])))
        self.assertTrue(self.rules_checker.check_property_rules('x_none_delete', 'read', create_context(self.policy, ['admin', 'member'])))
        self.assertTrue(self.rules_checker.check_property_rules('x_none_delete', 'update', create_context(self.policy, ['admin', 'member'])))
        self.assertFalse(self.rules_checker.check_property_rules('x_none_delete', 'delete', create_context(self.policy, [''])))

    def test_check_return_first_match(self):
        self.rules_checker = property_utils.PropertyRules()
        self.assertFalse(self.rules_checker.check_property_rules('x_foo_matcher', 'create', create_context(self.policy, [''])))
        self.assertFalse(self.rules_checker.check_property_rules('x_foo_matcher', 'read', create_context(self.policy, [''])))
        self.assertFalse(self.rules_checker.check_property_rules('x_foo_matcher', 'update', create_context(self.policy, [''])))
        self.assertFalse(self.rules_checker.check_property_rules('x_foo_matcher', 'delete', create_context(self.policy, [''])))