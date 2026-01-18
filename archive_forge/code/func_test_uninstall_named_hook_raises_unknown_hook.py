from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_uninstall_named_hook_raises_unknown_hook(self):
    hooks = Hooks('breezy.tests.test_hooks', 'some_hooks')
    self.assertRaises(UnknownHook, hooks.uninstall_named_hook, 'silly', '')