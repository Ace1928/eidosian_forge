import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
def test_reference_held_when_dispatching(self):
    dummy = DummyObservable()

    def event_factory(*args, **kwargs):
        nonlocal dummy
        del dummy
    notifier = create_notifier(handler=dummy.handler, event_factory=event_factory)
    notifier.add_to(dummy)
    notifier(a=1, b=2)