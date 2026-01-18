import os
import sys
from breezy import branch, osutils, registry, tests
def test_register_override(self):
    a_registry = registry.Registry()
    a_registry.register('one', 'one')
    self.assertRaises(KeyError, a_registry.register, 'one', 'two')
    self.assertRaises(KeyError, a_registry.register, 'one', 'two', override_existing=False)
    a_registry.register('one', 'two', override_existing=True)
    self.assertEqual('two', a_registry.get('one'))
    self.assertRaises(KeyError, a_registry.register_lazy, 'one', 'three', 'four')
    a_registry.register_lazy('one', 'module', 'member', override_existing=True)