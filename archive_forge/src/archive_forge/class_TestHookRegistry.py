from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
class TestHookRegistry(tests.TestCase):

    def test_items_are_reasonable_keys(self):
        for key, factory in known_hooks.items():
            self.assertTrue(callable(factory), 'The factory({!r}) for {!r} is not callable'.format(factory, key))
            obj = known_hooks_key_to_object(key)
            self.assertIsInstance(obj, Hooks)
            new_hooks = factory()
            self.assertIsInstance(obj, Hooks)
            self.assertEqual(type(obj), type(new_hooks))
            self.assertEqual('No hook name', new_hooks.get_hook_name(None))

    def test_known_hooks_key_to_object(self):
        self.assertIs(branch.Branch.hooks, known_hooks_key_to_object(('breezy.branch', 'Branch.hooks')))

    def test_known_hooks_key_to_parent_and_attribute(self):
        self.assertEqual((branch.Branch, 'hooks'), known_hooks.key_to_parent_and_attribute(('breezy.branch', 'Branch.hooks')))
        self.assertEqual((branch, 'Branch'), known_hooks.key_to_parent_and_attribute(('breezy.branch', 'Branch')))