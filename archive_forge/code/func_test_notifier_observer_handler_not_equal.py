import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def test_notifier_observer_handler_not_equal(self):
    handler = mock.Mock()
    graph = mock.Mock()
    target = mock.Mock()
    notifier1 = create_notifier(observer_handler=mock.Mock(), handler=handler, graph=graph, target=target, dispatcher=dispatch_here)
    notifier2 = create_notifier(observer_handler=mock.Mock(), handler=handler, graph=graph, target=target, dispatcher=dispatch_here)
    self.assertFalse(notifier1.equals(notifier2), 'Expected notifier1 to see notifier2 as different.')
    self.assertFalse(notifier2.equals(notifier1), 'Expected notifier2 to see notifier1 as different.')