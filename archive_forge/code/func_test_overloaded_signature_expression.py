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
def test_overloaded_signature_expression(self):
    expressions = [trait('age'), 'age', [trait('age')], ['age']]
    for expression in expressions:

        class NewPerson(Person):
            events = List()

            @observe(expression)
            def handler(self, event):
                self.events.append(event)
        person = NewPerson()
        person.age += 1
        self.assertEqual(len(person.events), 1)