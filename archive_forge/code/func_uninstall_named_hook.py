from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def uninstall_named_hook(self, hook_name, label):
    """Uninstall named hooks.

        :param hook_name: Hook point name
        :param label: Label of the callable to uninstall
        """
    try:
        hook = self[hook_name]
    except KeyError:
        raise UnknownHook(self.__class__.__name__, hook_name)
    try:
        uninstall = getattr(hook, 'uninstall')
    except AttributeError:
        raise errors.UnsupportedOperation(self.uninstall_named_hook, self)
    else:
        uninstall(label)