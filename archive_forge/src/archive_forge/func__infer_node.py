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
@_limit_value_infers
def _infer_node(context, element):
    debug.dbg('infer_node %s@%s in %s', element, element.start_pos, context)
    inference_state = context.inference_state
    typ = element.type
    if typ in ('name', 'number', 'string', 'atom', 'strings', 'keyword', 'fstring'):
        return infer_atom(context, element)
    elif typ == 'lambdef':
        return ValueSet([FunctionValue.from_context(context, element)])
    elif typ == 'expr_stmt':
        return infer_expr_stmt(context, element)
    elif typ in ('power', 'atom_expr'):
        first_child = element.children[0]
        children = element.children[1:]
        had_await = False
        if first_child.type == 'keyword' and first_child.value == 'await':
            had_await = True
            first_child = children.pop(0)
        value_set = context.infer_node(first_child)
        for i, trailer in enumerate(children):
            if trailer == '**':
                right = context.infer_node(children[i + 1])
                value_set = _infer_comparison(context, value_set, trailer, right)
                break
            value_set = infer_trailer(context, value_set, trailer)
        if had_await:
            return value_set.py__await__().py__stop_iteration_returns()
        return value_set
    elif typ in ('testlist_star_expr', 'testlist'):
        return ValueSet([iterable.SequenceLiteralValue(inference_state, context, element)])
    elif typ in ('not_test', 'factor'):
        value_set = context.infer_node(element.children[-1])
        for operator in element.children[:-1]:
            value_set = infer_factor(value_set, operator)
        return value_set
    elif typ == 'test':
        return context.infer_node(element.children[0]) | context.infer_node(element.children[-1])
    elif typ == 'operator':
        if element.value != '...':
            origin = element.parent
            raise AssertionError('unhandled operator %s in %s ' % (repr(element.value), origin))
        return ValueSet([compiled.builtin_from_name(inference_state, 'Ellipsis')])
    elif typ == 'dotted_name':
        value_set = infer_atom(context, element.children[0])
        for next_name in element.children[2::2]:
            value_set = value_set.py__getattribute__(next_name, name_context=context)
        return value_set
    elif typ == 'eval_input':
        return context.infer_node(element.children[0])
    elif typ == 'annassign':
        return annotation.infer_annotation(context, element.children[1]).execute_annotation()
    elif typ == 'yield_expr':
        if len(element.children) and element.children[1].type == 'yield_arg':
            element = element.children[1].children[1]
            generators = context.infer_node(element).py__getattribute__('__iter__').execute_with_values()
            return generators.py__stop_iteration_returns()
        return NO_VALUES
    elif typ == 'namedexpr_test':
        return context.infer_node(element.children[2])
    else:
        return infer_or_test(context, element)