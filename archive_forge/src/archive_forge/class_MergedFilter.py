from abc import abstractmethod
from typing import List, MutableMapping, Type
import weakref
from parso.tree import search_ancestor
from parso.python.tree import Name, UsedNamesMapping
from jedi.inference import flow_analysis
from jedi.inference.base_value import ValueSet, ValueWrapper, \
from jedi.parser_utils import get_cached_parent_scope, get_parso_cache_node
from jedi.inference.utils import to_list
from jedi.inference.names import TreeNameDefinition, ParamName, \
class MergedFilter:

    def __init__(self, *filters):
        self._filters = filters

    def get(self, name):
        return [n for filter in self._filters for n in filter.get(name)]

    def values(self):
        return [n for filter in self._filters for n in filter.values()]

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, ', '.join((str(f) for f in self._filters)))