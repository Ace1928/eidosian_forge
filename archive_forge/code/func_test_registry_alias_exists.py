import os
import sys
from breezy import branch, osutils, registry, tests
def test_registry_alias_exists(self):
    a_registry = registry.Registry()
    a_registry.register('one', 1, info='string info')
    a_registry.register('two', 2)
    self.assertRaises(KeyError, a_registry.register_alias, 'one', 'one')