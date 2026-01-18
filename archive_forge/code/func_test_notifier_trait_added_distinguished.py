import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_notifier_trait_added_distinguished(self):
    graph1 = create_graph(create_observer(name='some_value1', notify=True, optional=True))
    graph2 = create_graph(create_observer(name='some_value2', notify=True, optional=True))
    handler = mock.Mock()
    foo = ClassWithInstance()
    call_add_or_remove_notifiers(object=foo, graph=graph1, handler=handler, remove=False)
    call_add_or_remove_notifiers(object=foo, graph=graph2, handler=handler, remove=False)
    call_add_or_remove_notifiers(object=foo, graph=graph2, handler=handler, remove=True)
    foo.add_trait('some_value1', Int())
    foo.some_value1 += 1
    self.assertEqual(handler.call_count, 1)
    handler.reset_mock()
    foo.add_trait('some_value2', Int())
    foo.some_value2 += 1
    self.assertEqual(handler.call_count, 0)