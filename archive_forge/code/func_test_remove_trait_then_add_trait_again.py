import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_remove_trait_then_add_trait_again(self):
    graph = create_graph(create_observer(name='value1', notify=True, optional=False))
    handler = mock.Mock()
    foo = ClassWithTwoValue()
    call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler, remove=False)
    foo.value1 += 1
    handler.assert_called_once()
    handler.reset_mock()
    foo.remove_trait('value1')
    foo.value1 += 1
    handler.assert_not_called()
    foo.add_trait('value1', Int())
    foo.value1 += 1
    handler.assert_not_called()