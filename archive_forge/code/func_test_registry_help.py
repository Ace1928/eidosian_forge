import os
import sys
from breezy import branch, osutils, registry, tests
def test_registry_help(self):
    a_registry = registry.Registry()
    a_registry.register('one', 1, help='help text for one')
    a_registry.register_lazy('two', 'nonexistent_module', 'member', help='help text for two')
    help_calls = []

    def generic_help(reg, key):
        help_calls.append(key)
        return 'generic help for {}'.format(key)
    a_registry.register('three', 3, help=generic_help)
    a_registry.register_lazy('four', 'nonexistent_module', 'member2', help=generic_help)
    a_registry.register('five', 5)

    def help_from_object(reg, key):
        obj = reg.get(key)
        return obj.help()

    class SimpleObj:

        def help(self):
            return 'this is my help'
    a_registry.register('six', SimpleObj(), help=help_from_object)
    self.assertEqual('help text for one', a_registry.get_help('one'))
    self.assertEqual('help text for two', a_registry.get_help('two'))
    self.assertEqual('generic help for three', a_registry.get_help('three'))
    self.assertEqual(['three'], help_calls)
    self.assertEqual('generic help for four', a_registry.get_help('four'))
    self.assertEqual(['three', 'four'], help_calls)
    self.assertEqual(None, a_registry.get_help('five'))
    self.assertEqual('this is my help', a_registry.get_help('six'))
    self.assertRaises(KeyError, a_registry.get_help, None)
    self.assertRaises(KeyError, a_registry.get_help, 'seven')
    a_registry.default_key = 'one'
    self.assertEqual('help text for one', a_registry.get_help(None))
    self.assertRaises(KeyError, a_registry.get_help, 'seven')
    self.assertEqual([('five', None), ('four', 'generic help for four'), ('one', 'help text for one'), ('six', 'this is my help'), ('three', 'generic help for three'), ('two', 'help text for two')], sorted(((key, a_registry.get_help(key)) for key in a_registry.keys())))
    self.assertEqual(['four', 'four', 'three', 'three'], sorted(help_calls))