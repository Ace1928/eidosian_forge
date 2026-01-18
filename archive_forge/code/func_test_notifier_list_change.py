import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
def test_notifier_list_change(self):
    instance = ClassWithList(values=[])
    graph = create_graph(ListItemObserver(notify=True, optional=False))
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=instance.values, graph=graph, handler=handler)
    instance.values.append(1)
    ((event,), _), = handler.call_args_list
    self.assertIs(event.object, instance.values)
    self.assertEqual(event.added, [1])
    self.assertEqual(event.removed, [])
    self.assertEqual(event.index, 0)