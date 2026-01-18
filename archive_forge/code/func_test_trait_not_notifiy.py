import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
def test_trait_not_notifiy(self):
    actual = parse('a:b')
    expected = trait('a', notify=False).trait('b')
    self.assertEqual(actual, expected)