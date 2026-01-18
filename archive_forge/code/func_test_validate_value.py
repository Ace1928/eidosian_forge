import unittest
from unittest import mock
from traits.trait_types import Any, Dict, Event, Str, TraitDictObject
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_errors import TraitError
def test_validate_value(self):

    class Foo(HasTraits):
        mapping = Dict(Any, Str)
    foo = Foo(mapping={})
    foo.mapping['a'] = '1'
    with self.assertRaises(TraitError):
        foo.mapping['a'] = 1