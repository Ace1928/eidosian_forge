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
class TestObserverExpressionAnytrait(unittest.TestCase):
    """ Test anytrait function and method. """

    def test_anytrait_function_notify_true(self):
        expr = expression.anytrait(notify=True)
        expected = [create_graph(FilteredTraitObserver(filter=anytrait_filter, notify=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_anytrait_function_notify_false(self):
        expr = expression.anytrait(notify=False)
        expected = [create_graph(FilteredTraitObserver(filter=anytrait_filter, notify=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_anytrait_method_notify_true(self):
        expr = expression.trait('name').anytrait(notify=True)
        expected = [create_graph(NamedTraitObserver(name='name', notify=True, optional=False), FilteredTraitObserver(filter=anytrait_filter, notify=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_anytrait_method_notify_false(self):
        expr = expression.trait('name').anytrait(notify=False)
        expected = [create_graph(NamedTraitObserver(name='name', notify=True, optional=False), FilteredTraitObserver(filter=anytrait_filter, notify=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)