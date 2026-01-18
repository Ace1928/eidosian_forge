import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._set_item_observer import SetItemObserver
from traits.observation._testing import (
from traits.trait_set_object import TraitSet
from traits.trait_types import Set
def test_trait_set_iter_observables(self):
    instance = ClassWithSet()
    observer = create_observer(optional=False)
    actual_item, = list(observer.iter_observables(instance.values))
    self.assertIs(actual_item, instance.values)