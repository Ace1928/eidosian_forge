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
class _AbstractUsedNamesFilter(AbstractFilter):
    name_class = TreeNameDefinition

    def __init__(self, parent_context, node_context=None):
        if node_context is None:
            node_context = parent_context
        self._node_context = node_context
        self._parser_scope = node_context.tree_node
        module_context = node_context.get_root_context()
        path = module_context.py__file__()
        if path is None:
            self._parso_cache_node = None
        else:
            self._parso_cache_node = get_parso_cache_node(module_context.inference_state.latest_grammar if module_context.is_stub() else module_context.inference_state.grammar, path)
        self._used_names = module_context.tree_node.get_used_names()
        self.parent_context = parent_context

    def get(self, name):
        return self._convert_names(self._filter(_get_definition_names(self._parso_cache_node, self._used_names, name)))

    def _convert_names(self, names):
        return [self.name_class(self.parent_context, name) for name in names]

    def values(self):
        return self._convert_names((name for name_key in self._used_names for name in self._filter(_get_definition_names(self._parso_cache_node, self._used_names, name_key))))

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.parent_context)