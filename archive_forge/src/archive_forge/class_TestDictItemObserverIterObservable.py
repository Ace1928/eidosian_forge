import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._testing import (
from traits.trait_dict_object import TraitDict
from traits.trait_types import Dict, Str
class TestDictItemObserverIterObservable(unittest.TestCase):
    """ Test DictItemObserver.iter_observables """

    def test_trait_dict_iter_observables(self):
        instance = ClassWithDict()
        observer = create_observer(optional=False)
        actual_item, = list(observer.iter_observables(instance.values))
        self.assertIs(actual_item, instance.values)

    def test_dict_but_not_a_trait_dict(self):
        observer = create_observer(optional=False)
        with self.assertRaises(ValueError) as exception_context:
            list(observer.iter_observables(CustomDict()))
        self.assertIn('Expected a TraitDict to be observed, got', str(exception_context.exception))

    def test_custom_trait_dict_is_observable(self):
        observer = create_observer(optional=False)
        custom_trait_dict = CustomTraitDict()
        actual_item, = list(observer.iter_observables(custom_trait_dict))
        self.assertIs(actual_item, custom_trait_dict)

    def test_not_a_dict(self):
        observer = create_observer(optional=False)
        with self.assertRaises(ValueError) as exception_context:
            list(observer.iter_observables(None))
        self.assertIn('Expected a TraitDict to be observed, got', str(exception_context.exception))

    def test_optional_flag_not_a_dict(self):
        observer = create_observer(optional=True)
        actual = list(observer.iter_observables(None))
        self.assertEqual(actual, [])

    def test_optional_flag_not_an_observable(self):
        observer = create_observer(optional=True)
        actual = list(observer.iter_observables(CustomDict()))
        self.assertEqual(actual, [])