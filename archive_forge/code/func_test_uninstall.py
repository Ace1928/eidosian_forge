from .. import branch, errors
from .. import hooks as _mod_hooks
from .. import pyutils, tests
from ..hooks import (HookPoint, Hooks, UnknownHook, install_lazy_named_hook,
def test_uninstall(self):
    hook = HookPoint('foo', 'no docs', None, None)
    hook.hook_lazy('breezy.tests.test_hooks', 'TestHook.lazy_callback', 'my callback')
    self.assertEqual([TestHook.lazy_callback], list(hook))
    hook.uninstall('my callback')
    self.assertEqual([], list(hook))