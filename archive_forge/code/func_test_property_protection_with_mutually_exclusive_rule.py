from glance.api import policy
from glance.common import exception
from glance.common import property_utils
import glance.context
from glance.tests.unit import base
def test_property_protection_with_mutually_exclusive_rule(self):
    exclusive_rules = {'.*': {'create': ['@', '!'], 'read': ['fake-role'], 'update': ['fake-role'], 'delete': ['fake-role']}}
    self.set_property_protection_rules(exclusive_rules)
    self.assertRaises(exception.InvalidPropertyProtectionConfiguration, property_utils.PropertyRules)