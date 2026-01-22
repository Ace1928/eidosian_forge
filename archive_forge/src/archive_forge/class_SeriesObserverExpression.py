import functools
from traits.observation._anytrait_filter import anytrait_filter
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._observer_graph import ObserverGraph
from traits.observation._set_item_observer import SetItemObserver
class SeriesObserverExpression(ObserverExpression):
    """ Container of ObserverExpression for joining expressions in series.

    Parameters
    ----------
    first : ObserverExpression
        Left expression to be joined in series.
    second : ObserverExpression
        Right expression to be joined in series.
    """
    __slots__ = ('_first', '_second')

    def __init__(self, first, second):
        self._first = first
        self._second = second

    def __hash__(self):
        return hash((type(self).__name__, self._first, self._second))

    def __eq__(self, other):
        return type(self) is type(other) and self._first == other._first and (self._second == other._second)

    def _create_graphs(self, branches):
        branches = self._second._create_graphs(branches=branches)
        return self._first._create_graphs(branches=branches)