import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
class TestRestrictedNamedTraitObserverWithWrappedObserver(unittest.TestCase):
    """ Test the quantities inherited from the wrapped observer."""

    def test_notify_inherited(self):
        wrapped_observer = DummyObserver(notify=False)
        observer = _RestrictedNamedTraitObserver(name='name', wrapped_observer=wrapped_observer)
        self.assertEqual(observer.notify, wrapped_observer.notify)

    def test_notifier_inherited(self):
        notifier = DummyNotifier()
        wrapped_observer = DummyObserver(notifier=notifier)
        observer = _RestrictedNamedTraitObserver(name='name', wrapped_observer=wrapped_observer)
        self.assertEqual(observer.get_notifier(None, None, None), notifier)

    def test_maintainer_inherited(self):
        maintainer = DummyNotifier()
        wrapped_observer = DummyObserver(maintainer=maintainer)
        observer = _RestrictedNamedTraitObserver(name='name', wrapped_observer=wrapped_observer)
        self.assertEqual(observer.get_maintainer(None, None, None, None), maintainer)