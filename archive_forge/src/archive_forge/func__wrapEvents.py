import unittest
from zope.interface import Interface
from zope.interface.adapter import VerifyingAdapterRegistry
from zope.interface.registry import Components
def _wrapEvents(self):
    from zope.interface import registry
    _events = []

    def _notify(*args, **kw):
        _events.append((args, kw))
    _monkey = _Monkey(registry, notify=_notify)
    return (_monkey, _events)