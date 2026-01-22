from io import StringIO
from typing import Dict
from zope.interface import declarations, interface
from zope.interface.adapter import AdapterRegistry
from twisted.python import reflect
def _addHook(registry):
    """
    Add an adapter hook which will attempt to look up adapters in the given
    registry.

    @type registry: L{zope.interface.adapter.AdapterRegistry}

    @return: The hook which was added, for later use with L{_removeHook}.
    """
    lookup = registry.lookup1

    def _hook(iface, ob):
        factory = lookup(declarations.providedBy(ob), iface)
        if factory is None:
            return None
        else:
            return factory(ob)
    interface.adapter_hooks.append(_hook)
    return _hook