import inspect
import unittest
from traits.observation import expression
from traits.observation._anytrait_filter import anytrait_filter
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._set_item_observer import SetItemObserver
from traits.observation._observer_graph import ObserverGraph
class TestObserverExpressionFilter(unittest.TestCase):
    """ Test ObserverExpression.match """

    def setUp(self):

        def anytrait(name, trait):
            return True
        self.anytrait = anytrait

    def test_match_notify_true(self):
        expr = expression.match(filter=self.anytrait)
        expected = [create_graph(FilteredTraitObserver(filter=self.anytrait, notify=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_match_notify_false(self):
        expr = expression.match(filter=self.anytrait, notify=False)
        expected = [create_graph(FilteredTraitObserver(filter=self.anytrait, notify=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_match_method_notify_true(self):
        expr = expression.match(filter=self.anytrait).match(filter=self.anytrait)
        expected = [create_graph(FilteredTraitObserver(filter=self.anytrait, notify=True), FilteredTraitObserver(filter=self.anytrait, notify=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_match_method_notify_false(self):
        expr = expression.match(filter=self.anytrait).match(filter=self.anytrait, notify=False)
        expected = [create_graph(FilteredTraitObserver(filter=self.anytrait, notify=True), FilteredTraitObserver(filter=self.anytrait, notify=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_call_signatures(self):
        top_level = expression.match
        method = expression.ObserverExpression().match
        self.assertEqual(inspect.signature(top_level), inspect.signature(method))