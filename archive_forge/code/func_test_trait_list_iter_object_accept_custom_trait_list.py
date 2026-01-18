import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
def test_trait_list_iter_object_accept_custom_trait_list(self):
    instance = ClassWithList()
    instance.custom_trait_list = CustomTraitList([1, 2, 3])
    observer = ListItemObserver(notify=True, optional=False)
    actual = list(observer.iter_objects(instance.custom_trait_list))
    self.assertEqual(actual, [1, 2, 3])