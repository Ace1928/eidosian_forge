from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def _removeHook(hook):
    """
    Remove a previously added adapter hook.

    @param hook: An object previously returned by a call to L{_addHook}.  This
        will be removed from the list of adapter hooks.
    """
    interface.adapter_hooks.remove(hook)