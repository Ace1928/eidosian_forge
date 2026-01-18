import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
def test_trait_list_iter_objects(self):
    instance = ClassWithList()
    item1 = mock.Mock()
    item2 = mock.Mock()
    instance.values = [item1, item2]
    observer = ListItemObserver(notify=True, optional=False)
    actual = list(observer.iter_objects(instance.values))
    self.assertEqual(actual, [item1, item2])