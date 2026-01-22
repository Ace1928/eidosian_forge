import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
class TestObserverChangeNotifierCall(unittest.TestCase):
    """ Tests for the notifier being a callable."""

    def test_init_and_call(self):
        graph = mock.Mock()
        observer_handler = mock.Mock()
        event_factory = mock.Mock(return_value='Event')
        handler = mock.Mock()
        target = mock.Mock()
        dispatcher = mock.Mock()
        notifier = create_notifier(observer_handler=observer_handler, graph=graph, handler=handler, target=target, dispatcher=dispatcher, event_factory=event_factory)
        notifier(a=1, b=2)
        event_factory.assert_called_once_with(a=1, b=2)
        observer_handler.assert_called_once_with(event='Event', graph=graph, handler=handler, target=target, dispatcher=dispatcher)

    def test_call_with_prevent_event(self):
        observer_handler = mock.Mock()
        handler = mock.Mock()
        target = mock.Mock()
        notifier = create_notifier(observer_handler=observer_handler, handler=handler, target=target, event_factory=lambda value: value, prevent_event=lambda event: event != 'Fire')
        notifier('Hello')
        self.assertEqual(observer_handler.call_count, 0)
        notifier('Fire')
        self.assertEqual(observer_handler.call_count, 1)