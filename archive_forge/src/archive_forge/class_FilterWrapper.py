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
class FilterWrapper:
    name_wrapper_class: Type[NameWrapper]

    def __init__(self, wrapped_filter):
        self._wrapped_filter = wrapped_filter

    def wrap_names(self, names):
        return [self.name_wrapper_class(name) for name in names]

    def get(self, name):
        return self.wrap_names(self._wrapped_filter.get(name))

    def values(self):
        return self.wrap_names(self._wrapped_filter.values())