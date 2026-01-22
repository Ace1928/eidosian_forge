import unittest
from unittest import mock
from traits.api import HasTraits, Instance, Int, List
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._testing import (
from traits.trait_list_object import TraitList, TraitListObject
class TestListItemObserverIterObservable(unittest.TestCase):
    """ Test ListItemObserver.iter_observables """

    def test_trait_list_iter_observables(self):
        instance = ClassWithList()
        instance.values = [1, 2, 3]
        observer = ListItemObserver(notify=True, optional=False)
        actual_item, = list(observer.iter_observables(instance.values))
        self.assertIs(actual_item, instance.values)

    def test_trait_list_iter_observables_with_default_list(self):
        instance = ClassWithList()
        observer = ListItemObserver(notify=True, optional=False)
        actual_item, = list(observer.iter_observables(instance.values))
        self.assertIsInstance(actual_item, TraitListObject)

    def test_trait_list_iter_observables_accept_custom_trait_list(self):
        instance = ClassWithList()
        instance.custom_trait_list = CustomTraitList([1, 2, 3])
        observer = ListItemObserver(notify=True, optional=False)
        actual_item, = list(observer.iter_observables(instance.custom_trait_list))
        self.assertIs(actual_item, instance.custom_trait_list)

    def test_trait_list_iter_observables_error(self):
        instance = ClassWithList()
        instance.not_a_trait_list = CustomList()
        observer = ListItemObserver(notify=True, optional=False)
        with self.assertRaises(ValueError) as exception_context:
            next(observer.iter_observables(instance.not_a_trait_list))
        self.assertIn('Expected a TraitList to be observed', str(exception_context.exception))

    def test_trait_list_iter_observables_not_a_trait_list_optional(self):
        instance = ClassWithList()
        observer = ListItemObserver(notify=True, optional=True)
        self.assertIsNone(instance.not_a_trait_list)
        actual = list(observer.iter_observables(instance.not_a_trait_list))
        self.assertEqual(actual, [])
        instance.not_a_trait_list = CustomList()
        actual = list(observer.iter_observables(instance.not_a_trait_list))
        self.assertEqual(actual, [])

    def test_trait_list_iter_observables_not_a_list_error(self):
        instance = ClassWithList()
        observer = ListItemObserver(notify=True, optional=False)
        with self.assertRaises(ValueError) as exception_context:
            list(observer.iter_observables(instance.number))
        self.assertIn('Expected a TraitList to be observed', str(exception_context.exception))