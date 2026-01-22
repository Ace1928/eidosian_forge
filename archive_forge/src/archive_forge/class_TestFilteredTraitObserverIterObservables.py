import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.trait_base import Undefined, Uninitialized
from traits.trait_types import Float, Instance, Int, List
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._testing import (
class TestFilteredTraitObserverIterObservables(unittest.TestCase):
    """ Test FilteredTraitObserver.iter_observables """

    def test_iter_observables_with_filter(self):
        instance = DummyParent()
        observer = create_observer(filter=lambda name, trait: type(trait.trait_type) is Int)
        actual = list(observer.iter_observables(instance))
        expected = [instance._trait('number', 2), instance._trait('number2', 2)]
        self.assertCountEqual(actual, expected)