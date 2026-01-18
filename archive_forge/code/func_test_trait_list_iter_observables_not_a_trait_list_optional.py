import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
def test_trait_list_iter_observables_not_a_trait_list_optional(self):
    instance = ClassWithList()
    observer = ListItemObserver(notify=True, optional=True)
    self.assertIsNone(instance.not_a_trait_list)
    actual = list(observer.iter_observables(instance.not_a_trait_list))
    self.assertEqual(actual, [])
    instance.not_a_trait_list = CustomList()
    actual = list(observer.iter_observables(instance.not_a_trait_list))
    self.assertEqual(actual, [])