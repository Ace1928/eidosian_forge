import unittest
from unittest import mock
from traits.has_traits import HasTraits
from traits.observation._testing import (
from traits.observation._trait_added_observer import (
from traits.trait_types import Str
class TestTraitAddedObserverIterObservables(unittest.TestCase):
    """ Test sanity checks in iter_observables. """

    def test_iter_observables_get_trait_added_ctrait(self):
        observer = create_observer()
        instance = DummyHasTraitsClass()
        actual, = list(observer.iter_observables(instance))
        self.assertEqual(actual, instance._trait('trait_added', 2))

    def test_iter_observables_ignore_incompatible_object_if_optional(self):
        observer = create_observer(optional=True)
        actual = list(observer.iter_observables(None))
        self.assertEqual(actual, [])

    def test_iter_observables_error_incompatible_object_if_required(self):
        observer = create_observer(optional=False)
        with self.assertRaises(ValueError) as exception_cm:
            list(observer.iter_observables(None))
        self.assertIn("Unable to observe 'trait_added'", str(exception_cm.exception))