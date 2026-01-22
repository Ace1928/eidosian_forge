import unittest
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._testing import create_graph
from traits.observation.parsing import compile_str, parse
from traits.observation.expression import (
class TestParsingItems(unittest.TestCase):

    def test_items(self):
        actual = parse('items')
        expected = trait('items', optional=True) | dict_items(optional=True) | list_items(optional=True) | set_items(optional=True)
        self.assertEqual(actual, expected)

    def test_items_not_notify(self):
        actual = parse('items:attr')
        expected = (trait('items', notify=False, optional=True) | dict_items(notify=False, optional=True) | list_items(notify=False, optional=True) | set_items(notify=False, optional=True)).trait('attr')
        self.assertEqual(actual, expected)