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
class TestObserverExpressionTrait(unittest.TestCase):
    """ Test ObserverExpression.trait """

    def test_trait_name(self):
        expr = expression.trait('name')
        expected = [create_graph(NamedTraitObserver(name='name', notify=True, optional=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_trait_name_notify_false(self):
        expr = expression.trait('name', notify=False)
        expected = [create_graph(NamedTraitObserver(name='name', notify=False, optional=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_trait_name_optional_true(self):
        expr = expression.trait('name', optional=True)
        expected = [create_graph(NamedTraitObserver(name='name', notify=True, optional=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_trait_method(self):
        expr = expression.trait('name').trait('attr')
        expected = [create_graph(NamedTraitObserver(name='name', notify=True, optional=False), NamedTraitObserver(name='attr', notify=True, optional=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_trait_method_notify_false(self):
        expr = expression.trait('name').trait('attr', notify=False)
        expected = [create_graph(NamedTraitObserver(name='name', notify=True, optional=False), NamedTraitObserver(name='attr', notify=False, optional=False))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_trait_method_optional_true(self):
        expr = expression.trait('name').trait('attr', optional=True)
        expected = [create_graph(NamedTraitObserver(name='name', notify=True, optional=False), NamedTraitObserver(name='attr', notify=True, optional=True))]
        actual = expr._as_graphs()
        self.assertEqual(actual, expected)

    def test_call_signatures(self):
        top_level_trait = expression.trait
        method_trait = expression.ObserverExpression().trait
        self.assertEqual(inspect.signature(top_level_trait), inspect.signature(method_trait))