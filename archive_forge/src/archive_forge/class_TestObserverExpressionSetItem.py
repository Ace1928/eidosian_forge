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
class TestObserverExpressionSetItem(unittest.TestCase):
    """ Test ObserverExpression.set_items """

    def test_set_items(self):
        expr = expression.set_items()
        expected = [create_graph(SetItemObserver(notify=True, optional=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_set_items_notify_false(self):
        expr = expression.set_items(notify=False)
        expected = [create_graph(SetItemObserver(notify=False, optional=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_set_items_optional_true(self):
        expr = expression.set_items(optional=True)
        expected = [create_graph(SetItemObserver(notify=True, optional=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_set_items_method_notify(self):
        expr = expression.set_items().set_items(notify=False)
        expected = [create_graph(SetItemObserver(notify=True, optional=False), SetItemObserver(notify=False, optional=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_set_items_method_optional(self):
        expr = expression.set_items().set_items(optional=True)
        expected = [create_graph(SetItemObserver(notify=True, optional=False), SetItemObserver(notify=True, optional=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_call_signatures(self):
        top_level = expression.set_items
        method = expression.ObserverExpression().set_items
        self.assertEqual(inspect.signature(top_level), inspect.signature(method))