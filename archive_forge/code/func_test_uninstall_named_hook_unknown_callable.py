from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_uninstall_named_hook_unknown_callable(self):
    hooks = Hooks('breezy.tests.test_hooks', 'some_hooks')
    hooks.add_hook('set_rh', 'Set revision hsitory', (2, 0))
    self.assertRaises(KeyError, hooks.uninstall_named_hook, 'set_rh', 'demo')