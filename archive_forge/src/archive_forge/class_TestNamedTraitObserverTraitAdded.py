import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
class TestNamedTraitObserverTraitAdded(unittest.TestCase):
    """ Test integration with the trait_added event."""

    def test_observe_respond_to_trait_added(self):
        graph = create_graph(create_observer(name='value', notify=True, optional=True))
        handler = mock.Mock()
        foo = ClassWithInstance()
        call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler)
        foo.add_trait('value', Int())
        self.assertEqual(handler.call_count, 0)
        foo.value += 1
        self.assertEqual(handler.call_count, 1)

    def test_observe_remove_notifiers_remove_trait_added(self):
        graph = create_graph(create_observer(name='value', notify=True, optional=True))
        handler = mock.Mock()
        foo = ClassWithInstance()
        call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler, remove=False)
        call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler, remove=True)
        foo.add_trait('value', Int())
        self.assertEqual(handler.call_count, 0)
        foo.value += 1
        self.assertEqual(handler.call_count, 0)

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

    def test_add_trait_remove_trait_then_add_trait_again(self):
        graph = create_graph(create_observer(name='new_value', notify=True, optional=True))
        handler = mock.Mock()
        foo = ClassWithInstance()
        call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler, remove=False)
        foo.add_trait('new_value', Int())
        foo.new_value += 1
        handler.assert_called_once()
        handler.reset_mock()
        foo.remove_trait('new_value')
        foo.add_trait('new_value', Int())
        foo.new_value += 1
        handler.assert_called_once()

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

    def test_optional_trait_added(self):
        graph = create_graph(create_observer(name='value', notify=True, optional=True))
        handler = mock.Mock()
        not_an_has_traits_instance = mock.Mock()
        try:
            call_add_or_remove_notifiers(object=not_an_has_traits_instance, graph=graph, handler=handler)
        except Exception:
            self.fail('Optional flag should have been propagated.')