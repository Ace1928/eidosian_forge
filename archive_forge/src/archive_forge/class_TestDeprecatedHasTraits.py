import copy
import pickle
import unittest
from traits.has_traits import (
from traits.ctrait import CTrait
from traits.observation.api import (
from traits.observation.exception_handling import (
from traits.traits import ForwardProperty, generic_trait
from traits.trait_types import Event, Float, Instance, Int, List, Map, Str
from traits.trait_errors import TraitError
class TestDeprecatedHasTraits(unittest.TestCase):

    def test_deprecated(self):

        class TestSingletonHasTraits(SingletonHasTraits):
            pass

        class TestSingletonHasStrictTraits(SingletonHasStrictTraits):
            pass

        class TestSingletonHasPrivateTraits(SingletonHasPrivateTraits):
            pass
        with self.assertWarns(DeprecationWarning):
            TestSingletonHasTraits()
        with self.assertWarns(DeprecationWarning):
            TestSingletonHasStrictTraits()
        with self.assertWarns(DeprecationWarning):
            TestSingletonHasPrivateTraits()