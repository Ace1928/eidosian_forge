import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
class TestParsingAnytrait(unittest.TestCase):

    def test_anytrait(self):
        actual = parse('*')
        expected = anytrait()
        self.assertEqual(actual, expected)

    def test_trait_anytrait_not_notify(self):
        actual = parse('name:*')
        expected = trait('name', notify=False).anytrait()
        self.assertEqual(actual, expected)

    def test_anytrait_in_parallel_branch(self):
        actual = parse('a:*,b')
        expected = trait('a', notify=False).anytrait() | trait('b')
        self.assertEqual(actual, expected)

    def test_anytrait_in_invalid_position(self):
        invalid_expressions = ['*.*', '*:*', '*.name', '*.items', '*:name', '*.a,b', '[a.*,b].c']
        for expression in invalid_expressions:
            with self.subTest(expression=expression):
                with self.assertRaises(ValueError):
                    parse(expression)