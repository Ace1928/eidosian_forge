import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
def test_trait_list_iter_objects_ignore_if_optional_and_not_list(self):
    observer = ListItemObserver(notify=True, optional=True)
    actual = list(observer.iter_objects(set([1])))
    self.assertEqual(actual, [])