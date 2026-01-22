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
@debug.increase_indent
def _infer_expr_stmt(context, stmt, seek_name=None):
    """
    The starting point of the completion. A statement always owns a call
    list, which are the calls, that a statement does. In case multiple
    names are defined in the statement, `seek_name` returns the result for
    this name.

    expr_stmt: testlist_star_expr (annassign | augassign (yield_expr|testlist) |
                     ('=' (yield_expr|testlist_star_expr))*)
    annassign: ':' test ['=' test]
    augassign: ('+=' | '-=' | '*=' | '@=' | '/=' | '%=' | '&=' | '|=' | '^=' |
                '<<=' | '>>=' | '**=' | '//=')

    :param stmt: A `tree.ExprStmt`.
    """

    def check_setitem(stmt):
        atom_expr = stmt.children[0]
        if atom_expr.type not in ('atom_expr', 'power'):
            return (False, None)
        name = atom_expr.children[0]
        if name.type != 'name' or len(atom_expr.children) != 2:
            return (False, None)
        trailer = atom_expr.children[-1]
        return (trailer.children[0] == '[', trailer.children[1])
    debug.dbg('infer_expr_stmt %s (%s)', stmt, seek_name)
    rhs = stmt.get_rhs()
    value_set = context.infer_node(rhs)
    if seek_name:
        n = TreeNameDefinition(context, seek_name)
        value_set = check_tuple_assignments(n, value_set)
    first_operator = next(stmt.yield_operators(), None)
    is_setitem, subscriptlist = check_setitem(stmt)
    is_annassign = first_operator not in ('=', None) and first_operator.type == 'operator'
    if is_annassign or is_setitem:
        name = stmt.get_defined_names(include_setitem=True)[0].value
        left_values = context.py__getattribute__(name, position=stmt.start_pos)
        if is_setitem:

            def to_mod(v):
                c = ContextualizedSubscriptListNode(context, subscriptlist)
                if v.array_type == 'dict':
                    return DictModification(v, value_set, c)
                elif v.array_type == 'list':
                    return ListModification(v, value_set, c)
                return v
            value_set = ValueSet((to_mod(v) for v in left_values))
        else:
            operator = copy.copy(first_operator)
            operator.value = operator.value[:-1]
            for_stmt = tree.search_ancestor(stmt, 'for_stmt')
            if for_stmt is not None and for_stmt.type == 'for_stmt' and value_set and parser_utils.for_stmt_defines_one_name(for_stmt):
                node = for_stmt.get_testlist()
                cn = ContextualizedNode(context, node)
                ordered = list(cn.infer().iterate(cn))
                for lazy_value in ordered:
                    dct = {for_stmt.children[1].value: lazy_value.infer()}
                    with context.predefine_names(for_stmt, dct):
                        t = context.infer_node(rhs)
                        left_values = _infer_comparison(context, left_values, operator, t)
                value_set = left_values
            else:
                value_set = _infer_comparison(context, left_values, operator, value_set)
    debug.dbg('infer_expr_stmt result %s', value_set)
    return value_set