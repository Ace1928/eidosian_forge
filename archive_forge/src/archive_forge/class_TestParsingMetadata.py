import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
class TestParsingMetadata(unittest.TestCase):

    def test_metadata(self):
        actual = parse('+name')
        expected = metadata('name', notify=True)
        self.assertEqual(actual, expected)

    def test_metadata_notify_false(self):
        actual = parse('+name:+attr')
        expected = metadata('name', notify=False).metadata('attr', notify=True)
        self.assertEqual(actual, expected)