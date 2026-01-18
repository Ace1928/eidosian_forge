import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
def test_trait_list_iter_observables_with_default_list(self):
    instance = ClassWithList()
    observer = ListItemObserver(notify=True, optional=False)
    actual_item, = list(observer.iter_observables(instance.values))
    self.assertIsInstance(actual_item, TraitListObject)