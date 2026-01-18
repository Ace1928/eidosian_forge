import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
def test_notifier_custom_trait_list_change(self):
    instance = ClassWithList()
    instance.custom_trait_list = CustomTraitList()
    graph = create_graph(ListItemObserver(notify=True, optional=False))
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=instance.custom_trait_list, graph=graph, handler=handler)
    instance.custom_trait_list.append(1)
    ((event,), _), = handler.call_args_list
    self.assertIs(event.object, instance.custom_trait_list)
    self.assertEqual(event.added, [1])
    self.assertEqual(event.removed, [])
    self.assertEqual(event.index, 0)