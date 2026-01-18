import os
import sys
from breezy import branch, osutils, registry, tests
def test_registry_info(self):
    a_registry = registry.Registry()
    a_registry.register('one', 1, info='string info')
    a_registry.register_lazy('two', 'nonexistent_module', 'member', info=2)
    a_registry.register('three', 3, info=['a', 'list'])
    obj = object()
    a_registry.register_lazy('four', 'nonexistent_module', 'member2', info=obj)
    a_registry.register('five', 5)
    self.assertEqual('string info', a_registry.get_info('one'))
    self.assertEqual(2, a_registry.get_info('two'))
    self.assertEqual(['a', 'list'], a_registry.get_info('three'))
    self.assertIs(obj, a_registry.get_info('four'))
    self.assertIs(None, a_registry.get_info('five'))
    self.assertRaises(KeyError, a_registry.get_info, None)
    self.assertRaises(KeyError, a_registry.get_info, 'six')
    a_registry.default_key = 'one'
    self.assertEqual('string info', a_registry.get_info(None))
    self.assertRaises(KeyError, a_registry.get_info, 'six')
    self.assertEqual([('five', None), ('four', obj), ('one', 'string info'), ('three', ['a', 'list']), ('two', 2)], sorted(((key, a_registry.get_info(key)) for key in a_registry.keys())))