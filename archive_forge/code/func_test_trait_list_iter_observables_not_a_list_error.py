import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
def test_trait_list_iter_observables_not_a_list_error(self):
    instance = ClassWithList()
    observer = ListItemObserver(notify=True, optional=False)
    with self.assertRaises(ValueError) as exception_context:
        list(observer.iter_observables(instance.number))
    self.assertIn('Expected a TraitList to be observed', str(exception_context.exception))