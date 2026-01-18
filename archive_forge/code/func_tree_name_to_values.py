import copy
import itertools
from parso.python import tree
from jedi import debug
from jedi import parser_utils
from jedi.inference.base_value import ValueSet, NO_VALUES, ContextualizedNode, \
from jedi.inference.lazy_value import LazyTreeValue
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import analysis
from jedi.inference import imports
from jedi.inference import arguments
from jedi.inference.value import ClassValue, FunctionValue
from jedi.inference.value import iterable
from jedi.inference.value.dynamic_arrays import ListModification, DictModification
from jedi.inference.value import TreeInstance
from jedi.inference.helpers import is_string, is_literal, is_number, \
from jedi.inference.compiled.access import COMPARISON_OPERATORS
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.gradual.stub_value import VersionInfo
from jedi.inference.gradual import annotation
from jedi.inference.names import TreeNameDefinition
from jedi.inference.context import CompForContext
from jedi.inference.value.decorator import Decoratee
from jedi.plugins import plugin_manager
@plugin_manager.decorate()
def tree_name_to_values(inference_state, context, tree_name):
    value_set = NO_VALUES
    module_node = context.get_root_context().tree_node
    if module_node is not None:
        names = module_node.get_used_names().get(tree_name.value, [])
        found_annotation = False
        for name in names:
            expr_stmt = name.parent
            if expr_stmt.type == 'expr_stmt' and expr_stmt.children[1].type == 'annassign':
                correct_scope = parser_utils.get_parent_scope(name) == context.tree_node
                if correct_scope:
                    found_annotation = True
                    value_set |= annotation.infer_annotation(context, expr_stmt.children[1].children[1]).execute_annotation()
        if found_annotation:
            return value_set
    types = []
    node = tree_name.get_definition(import_name_always=True, include_setitem=True)
    if node is None:
        node = tree_name.parent
        if node.type == 'global_stmt':
            c = context.create_context(tree_name)
            if c.is_module():
                return NO_VALUES
            filter = next(c.get_filters())
            names = filter.get(tree_name.value)
            return ValueSet.from_sets((name.infer() for name in names))
        elif node.type not in ('import_from', 'import_name'):
            c = context.create_context(tree_name)
            return infer_atom(c, tree_name)
    typ = node.type
    if typ == 'for_stmt':
        types = annotation.find_type_from_comment_hint_for(context, node, tree_name)
        if types:
            return types
    if typ == 'with_stmt':
        types = annotation.find_type_from_comment_hint_with(context, node, tree_name)
        if types:
            return types
    if typ in ('for_stmt', 'comp_for', 'sync_comp_for'):
        try:
            types = context.predefined_names[node][tree_name.value]
        except KeyError:
            cn = ContextualizedNode(context, node.children[3])
            for_types = iterate_values(cn.infer(), contextualized_node=cn, is_async=node.parent.type == 'async_stmt')
            n = TreeNameDefinition(context, tree_name)
            types = check_tuple_assignments(n, for_types)
    elif typ == 'expr_stmt':
        types = infer_expr_stmt(context, node, tree_name)
    elif typ == 'with_stmt':
        value_managers = context.infer_node(node.get_test_node_from_name(tree_name))
        if node.parent.type == 'async_stmt':
            enter_methods = value_managers.py__getattribute__('__aenter__')
            coro = enter_methods.execute_with_values()
            return coro.py__await__().py__stop_iteration_returns()
        enter_methods = value_managers.py__getattribute__('__enter__')
        return enter_methods.execute_with_values()
    elif typ in ('import_from', 'import_name'):
        types = imports.infer_import(context, tree_name)
    elif typ in ('funcdef', 'classdef'):
        types = _apply_decorators(context, node)
    elif typ == 'try_stmt':
        exceptions = context.infer_node(tree_name.get_previous_sibling().get_previous_sibling())
        types = exceptions.execute_with_values()
    elif typ == 'param':
        types = NO_VALUES
    elif typ == 'del_stmt':
        types = NO_VALUES
    elif typ == 'namedexpr_test':
        types = infer_node(context, node)
    else:
        raise ValueError('Should not happen. type: %s' % typ)
    return types