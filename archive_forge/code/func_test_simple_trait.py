import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_simple_trait(self):
    actual = parse('a')
    expected = trait('a')
    self.assertEqual(actual, expected)