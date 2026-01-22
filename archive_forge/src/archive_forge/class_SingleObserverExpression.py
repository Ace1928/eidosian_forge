import functools
from traits.observation._anytrait_filter import anytrait_filter
from traits.observation._dict_item_observer import DictItemObserver
from traits.observation._filtered_trait_observer import FilteredTraitObserver
from traits.observation._list_item_observer import ListItemObserver
from traits.observation._metadata_filter import MetadataFilter
from traits.observation._named_trait_observer import NamedTraitObserver
from traits.observation._observer_graph import ObserverGraph
from traits.observation._set_item_observer import SetItemObserver
class SingleObserverExpression(ObserverExpression):
    """ Container of ObserverExpression for wrapping a single observer.
    """
    __slots__ = ('_observer',)

    def __init__(self, observer):
        self._observer = observer

    def __hash__(self):
        return hash((type(self).__name__, self._observer))

    def __eq__(self, other):
        return type(self) is type(other) and self._observer == other._observer

    def _create_graphs(self, branches):
        return [ObserverGraph(node=self._observer, children=branches)]