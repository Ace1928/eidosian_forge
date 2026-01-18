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
def test_observe_post_init(self):

    class PersonWithPostInt(Person):
        events = List()

        @observe('age', post_init=True)
        def handler(self, event):
            self.events.append(event)
    person = PersonWithPostInt(age=10)
    self.assertEqual(len(person.events), 0)
    person.age += 1
    self.assertEqual(len(person.events), 1)