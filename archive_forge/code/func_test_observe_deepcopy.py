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
def test_observe_deepcopy(self):
    person = PersonWithObserve()
    copied = copy.deepcopy(person)
    copied.age += 1
    self.assertEqual(len(copied.events), 1)
    self.assertEqual(len(person.events), 0)