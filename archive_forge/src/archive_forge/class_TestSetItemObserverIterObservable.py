import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._set_item_observer import SetItemObserver
from traits.observation._testing import (
from traits.trait_set_object import TraitSet
from traits.trait_types import Set
class TestSetItemObserverIterObservable(unittest.TestCase):
    """ Test SetItemObserver.iter_observables """

    def test_trait_set_iter_observables(self):
        instance = ClassWithSet()
        observer = create_observer(optional=False)
        actual_item, = list(observer.iter_observables(instance.values))
        self.assertIs(actual_item, instance.values)

    def test_set_but_not_a_trait_set(self):
        observer = create_observer(optional=False)
        with self.assertRaises(ValueError) as exception_context:
            list(observer.iter_observables(CustomSet()))
        self.assertIn('Expected a TraitSet to be observed, got', str(exception_context.exception))

    def test_iter_observables_custom_trait_set(self):
        custom_trait_set = CustomTraitSet()
        observer = create_observer()
        actual_item, = list(observer.iter_observables(custom_trait_set))
        self.assertIs(actual_item, custom_trait_set)

    def test_not_a_set(self):
        observer = create_observer(optional=False)
        with self.assertRaises(ValueError) as exception_context:
            list(observer.iter_observables(None))
        self.assertIn('Expected a TraitSet to be observed, got', str(exception_context.exception))

    def test_optional_flag_not_a_set(self):
        observer = create_observer(optional=True)
        actual = list(observer.iter_observables(None))
        self.assertEqual(actual, [])

    def test_optional_flag_not_an_observable(self):
        observer = create_observer(optional=True)
        actual = list(observer.iter_observables(CustomSet()))
        self.assertEqual(actual, [])