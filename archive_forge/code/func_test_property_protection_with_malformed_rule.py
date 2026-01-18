from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_property_protection_with_malformed_rule(self):
    malformed_rules = {'^[0-9)': {'create': ['fake-policy'], 'read': ['fake-policy'], 'update': ['fake-policy'], 'delete': ['fake-policy']}}
    self.set_property_protection_rules(malformed_rules)
    self.assertRaises(exception.InvalidPropertyProtectionConfiguration, property_utils.PropertyRules)