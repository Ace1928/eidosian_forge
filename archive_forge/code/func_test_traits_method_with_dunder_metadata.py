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
def test_traits_method_with_dunder_metadata(self):

    class A(HasTraits):
        foo = Int(__extension_point__=True)
        bar = Int(__extension_point__=False)
        baz = Int()
    a = A(foo=3, bar=4, baz=5)
    self.assertEqual(a.traits(__extension_point__=True), {'foo': a.trait('foo')})
    self.assertEqual(A.class_traits(__extension_point__=True), {'foo': A.class_traits()['foo']})