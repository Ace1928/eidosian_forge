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
def test_notifiers_on_object(self):

    class Foo(HasTraits):
        x = Int()
    foo = Foo(x=1)
    self.assertEqual(foo._notifiers(True), [])

    def handler():
        pass
    foo.on_trait_change(handler, name='anytrait')
    notifiers = foo._notifiers(True)
    self.assertEqual(len(notifiers), 1)
    onotifier, = notifiers
    self.assertEqual(onotifier.handler, handler)