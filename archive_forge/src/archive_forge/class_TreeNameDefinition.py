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
class TreeNameDefinition(AbstractTreeName):
    _API_TYPES = dict(import_name='module', import_from='module', funcdef='function', param='param', classdef='class')

    def infer(self):
        from jedi.inference.syntax_tree import tree_name_to_values
        return tree_name_to_values(self.parent_context.inference_state, self.parent_context, self.tree_name)

    @property
    def api_type(self):
        definition = self.tree_name.get_definition(import_name_always=True)
        if definition is None:
            return 'statement'
        return self._API_TYPES.get(definition.type, 'statement')

    def assignment_indexes(self):
        """
        Returns an array of tuple(int, node) of the indexes that are used in
        tuple assignments.

        For example if the name is ``y`` in the following code::

            x, (y, z) = 2, ''

        would result in ``[(1, xyz_node), (0, yz_node)]``.

        When searching for b in the case ``a, *b, c = [...]`` it will return::

            [(slice(1, -1), abc_node)]
        """
        indexes = []
        is_star_expr = False
        node = self.tree_name.parent
        compare = self.tree_name
        while node is not None:
            if node.type in ('testlist', 'testlist_comp', 'testlist_star_expr', 'exprlist'):
                for i, child in enumerate(node.children):
                    if child == compare:
                        index = int(i / 2)
                        if is_star_expr:
                            from_end = int((len(node.children) - i) / 2)
                            index = slice(index, -from_end)
                        indexes.insert(0, (index, node))
                        break
                else:
                    raise LookupError("Couldn't find the assignment.")
                is_star_expr = False
            elif node.type == 'star_expr':
                is_star_expr = True
            elif node.type in ('expr_stmt', 'sync_comp_for'):
                break
            compare = node
            node = node.parent
        return indexes

    @property
    def inference_state(self):
        return self.parent_context.inference_state

    @inference_state_method_cache(default='')
    def py__doc__(self):
        api_type = self.api_type
        if api_type in ('function', 'class', 'property'):
            if self.parent_context.get_root_context().is_stub():
                from jedi.inference.gradual.conversion import convert_names
                names = convert_names([self], prefer_stub_to_compiled=False)
                if self not in names:
                    return _merge_name_docs(names)
            return clean_scope_docstring(self.tree_name.get_definition())
        if api_type == 'module':
            names = self.goto()
            if self not in names:
                return _merge_name_docs(names)
        if api_type == 'statement' and self.tree_name.is_definition():
            return find_statement_documentation(self.tree_name.get_definition())
        return ''