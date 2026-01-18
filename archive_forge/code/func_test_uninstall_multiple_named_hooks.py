from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_uninstall_multiple_named_hooks(self):
    hooks = Hooks('breezy.tests.test_hooks', 'some_hooks')
    hooks.add_hook('set_rh', 'Set revision history', (2, 0))
    hooks.install_named_hook('set_rh', 1, 'demo')
    hooks.install_named_hook('set_rh', 2, 'demo')
    hooks.install_named_hook('set_rh', 3, 'othername')
    self.assertEqual(3, len(hooks['set_rh']))
    hooks.uninstall_named_hook('set_rh', 'demo')
    self.assertEqual(1, len(hooks['set_rh']))