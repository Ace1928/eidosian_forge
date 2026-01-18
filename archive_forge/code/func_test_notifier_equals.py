import unittest
from unittest import mock
import weakref
from traits.api import HasTraits, Instance, Int
from traits.observation._observer_change_notifier import ObserverChangeNotifier
from traits.observation._observer_graph import ObserverGraph
from traits.observation.exceptions import NotifierNotFound
def test_notifier_equals(self):
    observer_handler = mock.Mock()
    handler = mock.Mock()
    graph = mock.Mock()
    target = mock.Mock()
    notifier1 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph, target=target, dispatcher=dispatch_here)
    notifier2 = create_notifier(observer_handler=observer_handler, handler=handler, graph=graph, target=target, dispatcher=dispatch_here)
    self.assertTrue(notifier1.equals(notifier2), 'Expected notifier1 to see notifier2 as equals.')
    self.assertTrue(notifier2.equals(notifier1), 'Expected notifier2 to see notifier1 as equals.')