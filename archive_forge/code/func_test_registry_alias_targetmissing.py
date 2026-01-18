import os
import sys
from breezy import branch, osutils, registry, tests
def test_registry_alias_targetmissing(self):
    a_registry = registry.Registry()
    self.assertRaises(KeyError, a_registry.register_alias, 'one', 'two')