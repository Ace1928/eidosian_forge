import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
def test_optional_observers(self):
    instance = ClassWithList()
    graph = create_graph(ListItemObserver(notify=True, optional=True))
    handler = mock.Mock()
    call_add_or_remove_notifiers(object=instance.not_a_trait_list, graph=graph, handler=handler)
    instance.not_a_trait_list = CustomList()
    instance.not_a_trait_list.append(1)
    self.assertEqual(handler.call_count, 0)