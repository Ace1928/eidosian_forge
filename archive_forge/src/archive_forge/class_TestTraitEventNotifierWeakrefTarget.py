import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
class TestTraitEventNotifierWeakrefTarget(unittest.TestCase):
    """ Test weakref handling for target in TraitEventNotifier."""

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

    def tearDown(self):
        pass

    def test_notifier_does_not_prevent_object_deletion(self):
        target = DummyObservable()
        target.internal_object = DummyObservable()
        target_ref = weakref.ref(target)
        notifier = create_notifier(target=target)
        notifier.add_to(target.internal_object)
        del target
        self.assertIsNone(target_ref())

    def test_callable_disabled_if_target_removed(self):
        target = mock.Mock()
        handler = mock.Mock()
        notifier = create_notifier(handler=handler, target=target)
        notifier(a=1, b=2)
        self.assertEqual(handler.call_count, 1)
        handler.reset_mock()
        del target
        notifier(a=1, b=2)
        handler.assert_not_called()