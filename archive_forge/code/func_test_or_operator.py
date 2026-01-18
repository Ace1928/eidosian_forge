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
def test_or_operator(self):
    observer1 = 1
    observer2 = 2
    expr1 = create_expression(observer1)
    expr2 = create_expression(observer2)
    expr = expr1 | expr2
    expected = [create_graph(observer1), create_graph(observer2)]
    actual = expr._as_graphs()
    self.assertEqual(actual, expected)