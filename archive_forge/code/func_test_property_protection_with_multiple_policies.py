from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_property_protection_with_multiple_policies(self):
    malformed_rules = {'^x_.*': {'create': ['fake-policy, another_pol'], 'read': ['fake-policy'], 'update': ['fake-policy'], 'delete': ['fake-policy']}}
    self.set_property_protection_rules(malformed_rules)
    self.assertRaises(exception.InvalidPropertyProtectionConfiguration, property_utils.PropertyRules)