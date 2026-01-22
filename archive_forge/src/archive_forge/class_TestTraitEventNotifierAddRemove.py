import unittest
from unittest import mock
import weakref
from traits.observation.exception_handling import (
from traits.observation.exceptions import NotifierNotFound
from traits.observation._trait_event_notifier import TraitEventNotifier
class TestTraitEventNotifierAddRemove(unittest.TestCase):
    """ Test TraitEventNotifier capability of adding/removing
    itself to/from an observable.
    """

    def setUp(self):
        push_exception_handler(reraise_exceptions=True)
        self.addCleanup(pop_exception_handler)

    def tearDown(self):
        pass

    def test_add_to_observable(self):
        dummy = DummyObservable()
        dummy.notifiers = [str, float]
        notifier = create_notifier()
        notifier.add_to(dummy)
        self.assertEqual(dummy.notifiers, [str, float, notifier])

    def test_add_to_observable_twice_increase_count(self):
        dummy = DummyObservable()

        def handler(event):
            pass
        notifier1 = create_notifier(handler=handler, target=_DUMMY_TARGET)
        notifier2 = create_notifier(handler=handler, target=_DUMMY_TARGET)
        notifier1.add_to(dummy)
        notifier2.add_to(dummy)
        self.assertEqual(dummy.notifiers, [notifier1])
        self.assertEqual(notifier1._ref_count, 2)

    def test_add_to_observable_different_notifier(self):
        dummy = DummyObservable()

        def handler(event):
            pass
        notifier1 = create_notifier(handler=handler, target=_DUMMY_TARGET)
        notifier2 = create_notifier(handler=handler, target=dummy)
        notifier1.add_to(dummy)
        notifier2.add_to(dummy)
        self.assertEqual(dummy.notifiers, [notifier1, notifier2])

    def test_remove_from_observable(self):
        dummy = DummyObservable()

        def handler(event):
            pass
        notifier1 = create_notifier(handler=handler, target=_DUMMY_TARGET)
        notifier2 = create_notifier(handler=handler, target=_DUMMY_TARGET)
        notifier1.add_to(dummy)
        self.assertEqual(dummy.notifiers, [notifier1])
        notifier2.remove_from(dummy)
        self.assertEqual(dummy.notifiers, [])

    def test_remove_from_observable_with_ref_count(self):
        dummy = DummyObservable()

        def handler(event):
            pass
        notifier1 = create_notifier(handler=handler, target=_DUMMY_TARGET)
        notifier2 = create_notifier(handler=handler, target=_DUMMY_TARGET)
        notifier1.add_to(dummy)
        notifier1.add_to(dummy)
        self.assertEqual(dummy.notifiers, [notifier1])
        notifier2.remove_from(dummy)
        self.assertEqual(dummy.notifiers, [notifier1])
        notifier2.remove_from(dummy)
        self.assertEqual(dummy.notifiers, [])

    def test_remove_from_error_if_not_found(self):
        dummy = DummyObservable()
        notifier1 = create_notifier()
        with self.assertRaises(NotifierNotFound) as e:
            notifier1.remove_from(dummy)
        self.assertEqual(str(e.exception), 'Notifier not found.')

    def test_remove_from_differentiate_not_equal_notifier(self):
        dummy = DummyObservable()
        notifier1 = create_notifier(handler=mock.Mock())
        notifier2 = create_notifier(handler=mock.Mock())
        notifier1.add_to(dummy)
        notifier2.add_to(dummy)
        notifier2.remove_from(dummy)
        self.assertEqual(dummy.notifiers, [notifier1])

    def test_add_to_multiple_observables(self):
        dummy1 = DummyObservable()
        dummy2 = DummyObservable()
        notifier = create_notifier()
        notifier.add_to(dummy1)
        with self.assertRaises(RuntimeError) as exception_context:
            notifier.add_to(dummy2)
        self.assertEqual(str(exception_context.exception), 'Sharing notifiers across observables is unexpected.')