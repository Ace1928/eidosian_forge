import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_remove_notifiers_after_trait_added(self):
    graph = create_graph(create_observer(name='value', notify=True, optional=True))
    handler = mock.Mock()
    foo = ClassWithInstance()
    call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler, remove=False)
    foo.add_trait('value', Int())
    foo.value += 1
    self.assertEqual(handler.call_count, 1)
    handler.reset_mock()
    call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler, remove=True)
    foo.value += 1
    self.assertEqual(handler.call_count, 0)