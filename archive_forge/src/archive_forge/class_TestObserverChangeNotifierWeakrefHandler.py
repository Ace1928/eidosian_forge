import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
class TestObserverChangeNotifierWeakrefHandler(unittest.TestCase):
    """ Test for using weak references when the user handler is a method
    of an instance.
    """

    def test_instance_can_be_garbage_collected(self):
        instance = DummyClass()
        instance_ref = weakref.ref(instance)
        notifier = create_notifier(handler=instance.dummy_method)
        del instance
        self.assertIsNone(instance_ref())

    def test_deleted_handler_silence_notifier(self):

        def observer_handler(*args, **kwargs):
            pass
        instance = DummyClass()
        method_ref = weakref.WeakMethod(instance.dummy_method)
        target = mock.Mock()
        event_factory = mock.Mock()
        notifier = create_notifier(observer_handler=observer_handler, target=target, handler=instance.dummy_method, event_factory=event_factory)
        notifier(b=3)
        self.assertEqual(event_factory.call_count, 1)
        event_factory.reset_mock()
        del instance
        self.assertIsNone(method_ref())
        notifier(a=1, b=2)
        event_factory.assert_not_called()