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
def test_trait_and_object_notifiers_called(self):
    side_effects = []

    class Foo(HasTraits):
        x = Int()
        y = Int()

        def _x_changed(self):
            side_effects.append('x')

    def object_handler():
        side_effects.append('object')
    foo = Foo()
    foo.on_trait_change(object_handler, name='anytrait')
    side_effects.clear()
    foo.x = 3
    self.assertEqual(side_effects, ['x', 'object'])
    side_effects.clear()
    foo.y = 4
    self.assertEqual(side_effects, ['object'])