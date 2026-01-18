import unittest
from unittest import mock
from traits.api import Bool, HasTraits, Int, Instance
from traits.observation._named_trait_observer import (
from traits.observation._observer_graph import ObserverGraph
from traits.observation._testing import (
def test_notifier_extended_trait_change(self):
    foo = ClassWithInstance()
    graph = create_graph(create_observer(name='instance', notify=True), create_observer(name='value1', notify=True))
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=foo, graph=graph, handler=handler)
    self.assertIsNone(foo.instance)
    foo.instance = ClassWithTwoValue()
    ((event,), _), = handler.call_args_list
    self.assertEqual(event.object, foo)
    self.assertEqual(event.name, 'instance')
    self.assertEqual(event.old, None)
    self.assertEqual(event.new, foo.instance)
    handler.reset_mock()
    foo.instance.value1 += 1
    ((event,), _), = handler.call_args_list
    self.assertEqual(event.object, foo.instance)
    self.assertEqual(event.name, 'value1')
    self.assertEqual(event.old, 0)
    self.assertEqual(event.new, 1)