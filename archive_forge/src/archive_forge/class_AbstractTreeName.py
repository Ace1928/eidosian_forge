from abc import abstractmethod
from inspect import Parameter
from typing import Optional, Tuple
from parso.tree import search_ancestor
from jedi.parser_utils import find_statement_documentation, clean_scope_docstring
from jedi.inference.utils import unite
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.cache import inference_state_method_cache
from jedi.inference import docstrings
from jedi.cache import memoize_method
from jedi.inference.helpers import deep_ast_copy, infer_call_of_leaf
from jedi.plugins import plugin_manager
class AbstractTreeName(AbstractNameDefinition):

    def __init__(self, parent_context, tree_name):
        self.parent_context = parent_context
        self.tree_name = tree_name

    def get_qualified_names(self, include_module_names=False):
        import_node = search_ancestor(self.tree_name, 'import_name', 'import_from')
        if import_node is not None and (not (import_node.level == 1 and self.get_root_context().get_value().is_package())):
            if include_module_names and (not import_node.level):
                return tuple((n.value for n in import_node.get_path_for_name(self.tree_name)))
            else:
                return None
        return super().get_qualified_names(include_module_names)

    def _get_qualified_names(self):
        parent_names = self.parent_context.get_qualified_names()
        if parent_names is None:
            return None
        return parent_names + (self.tree_name.value,)

    def get_defining_qualified_value(self):
        if self.is_import():
            raise NotImplementedError("Shouldn't really happen, please report")
        elif self.parent_context:
            return self.parent_context.get_value()
        return None

    def goto(self):
        context = self.parent_context
        name = self.tree_name
        definition = name.get_definition(import_name_always=True)
        if definition is not None:
            type_ = definition.type
            if type_ == 'expr_stmt':
                is_simple_name = name.parent.type not in ('power', 'trailer')
                if is_simple_name:
                    return [self]
            elif type_ in ('import_from', 'import_name'):
                from jedi.inference.imports import goto_import
                module_names = goto_import(context, name)
                return module_names
            else:
                return [self]
        else:
            from jedi.inference.imports import follow_error_node_imports_if_possible
            values = follow_error_node_imports_if_possible(context, name)
            if values is not None:
                return [value.name for value in values]
        par = name.parent
        node_type = par.type
        if node_type == 'argument' and par.children[1] == '=' and (par.children[0] == name):
            trailer = par.parent
            if trailer.type == 'arglist':
                trailer = trailer.parent
            if trailer.type != 'classdef':
                if trailer.type == 'decorator':
                    value_set = context.infer_node(trailer.children[1])
                else:
                    i = trailer.parent.children.index(trailer)
                    to_infer = trailer.parent.children[:i]
                    if to_infer[0] == 'await':
                        to_infer.pop(0)
                    value_set = context.infer_node(to_infer[0])
                    from jedi.inference.syntax_tree import infer_trailer
                    for trailer in to_infer[1:]:
                        value_set = infer_trailer(context, value_set, trailer)
                param_names = []
                for value in value_set:
                    for signature in value.get_signatures():
                        for param_name in signature.get_param_names():
                            if param_name.string_name == name.value:
                                param_names.append(param_name)
                return param_names
        elif node_type == 'dotted_name':
            index = par.children.index(name)
            if index > 0:
                new_dotted = deep_ast_copy(par)
                new_dotted.children[index - 1:] = []
                values = context.infer_node(new_dotted)
                return unite((value.goto(name, name_context=context) for value in values))
        if node_type == 'trailer' and par.children[0] == '.':
            values = infer_call_of_leaf(context, name, cut_own_trailer=True)
            return values.goto(name, name_context=context)
        else:
            stmt = search_ancestor(name, 'expr_stmt', 'lambdef') or name
            if stmt.type == 'lambdef':
                stmt = name
            return context.goto(name, position=stmt.start_pos)

    def is_import(self):
        imp = search_ancestor(self.tree_name, 'import_from', 'import_name')
        return imp is not None

    @property
    def string_name(self):
        return self.tree_name.value

    @property
    def start_pos(self):
        return self.tree_name.start_pos