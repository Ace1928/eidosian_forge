from abc import abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Optional
from parso.tree import search_ancestor
from parso.python.tree import Name
from jedi.inference.filters import ParserTreeFilter, MergedFilter, \
from jedi.inference.names import AnonymousParamName, TreeNameDefinition
from jedi.inference.base_value import NO_VALUES, ValueSet
from jedi.parser_utils import get_parent_scope
from jedi import debug
from jedi import parser_utils
class AbstractContext:

    def __init__(self, inference_state):
        self.inference_state = inference_state
        self.predefined_names = {}

    @abstractmethod
    def get_filters(self, until_position=None, origin_scope=None):
        raise NotImplementedError

    def goto(self, name_or_str, position):
        from jedi.inference import finder
        filters = _get_global_filters_for_name(self, name_or_str if isinstance(name_or_str, Name) else None, position)
        names = finder.filter_name(filters, name_or_str)
        debug.dbg('context.goto %s in (%s): %s', name_or_str, self, names)
        return names

    def py__getattribute__(self, name_or_str, name_context=None, position=None, analysis_errors=True):
        """
        :param position: Position of the last statement -> tuple of line, column
        """
        if name_context is None:
            name_context = self
        names = self.goto(name_or_str, position)
        string_name = name_or_str.value if isinstance(name_or_str, Name) else name_or_str
        found_predefined_types = None
        if self.predefined_names and isinstance(name_or_str, Name):
            node = name_or_str
            while node is not None and (not parser_utils.is_scope(node)):
                node = node.parent
                if node.type in ('if_stmt', 'for_stmt', 'comp_for', 'sync_comp_for'):
                    try:
                        name_dict = self.predefined_names[node]
                        types = name_dict[string_name]
                    except KeyError:
                        continue
                    else:
                        found_predefined_types = types
                        break
        if found_predefined_types is not None and names:
            from jedi.inference import flow_analysis
            check = flow_analysis.reachability_check(context=self, value_scope=self.tree_node, node=name_or_str)
            if check is flow_analysis.UNREACHABLE:
                values = NO_VALUES
            else:
                values = found_predefined_types
        else:
            values = ValueSet.from_sets((name.infer() for name in names))
        if not names and (not values) and analysis_errors:
            if isinstance(name_or_str, Name):
                from jedi.inference import analysis
                message = "NameError: name '%s' is not defined." % string_name
                analysis.add(name_context, 'name-error', name_or_str, message)
        debug.dbg('context.names_to_types: %s -> %s', names, values)
        if values:
            return values
        return self._check_for_additional_knowledge(name_or_str, name_context, position)

    def _check_for_additional_knowledge(self, name_or_str, name_context, position):
        name_context = name_context or self
        if isinstance(name_or_str, Name) and (not name_context.is_instance()):
            flow_scope = name_or_str
            base_nodes = [name_context.tree_node]
            if any((b.type in ('comp_for', 'sync_comp_for') for b in base_nodes)):
                return NO_VALUES
            from jedi.inference.finder import check_flow_information
            while True:
                flow_scope = get_parent_scope(flow_scope, include_flows=True)
                n = check_flow_information(name_context, flow_scope, name_or_str, position)
                if n is not None:
                    return n
                if flow_scope in base_nodes:
                    break
        return NO_VALUES

    def get_root_context(self):
        parent_context = self.parent_context
        if parent_context is None:
            return self
        return parent_context.get_root_context()

    def is_module(self):
        return False

    def is_builtins_module(self):
        return False

    def is_class(self):
        return False

    def is_stub(self):
        return False

    def is_instance(self):
        return False

    def is_compiled(self):
        return False

    def is_bound_method(self):
        return False

    @abstractmethod
    def py__name__(self):
        raise NotImplementedError

    def get_value(self):
        raise NotImplementedError

    @property
    def name(self):
        return None

    def get_qualified_names(self):
        return ()

    def py__doc__(self):
        return ''

    @contextmanager
    def predefine_names(self, flow_scope, dct):
        predefined = self.predefined_names
        predefined[flow_scope] = dct
        try:
            yield
        finally:
            del predefined[flow_scope]