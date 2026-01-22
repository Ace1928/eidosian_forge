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
@inference_state_method_cache()
def _apply_decorators(context, node):
    """
    Returns the function, that should to be executed in the end.
    This is also the places where the decorators are processed.
    """
    if node.type == 'classdef':
        decoratee_value = ClassValue(context.inference_state, parent_context=context, tree_node=node)
    else:
        decoratee_value = FunctionValue.from_context(context, node)
    initial = values = ValueSet([decoratee_value])
    if is_big_annoying_library(context):
        return values
    for dec in reversed(node.get_decorators()):
        debug.dbg('decorator: %s %s', dec, values, color='MAGENTA')
        with debug.increase_indent_cm():
            dec_values = context.infer_node(dec.children[1])
            trailer_nodes = dec.children[2:-1]
            if trailer_nodes:
                trailer = tree.PythonNode('trailer', trailer_nodes)
                trailer.parent = dec
                dec_values = infer_trailer(context, dec_values, trailer)
            if not len(dec_values):
                code = dec.get_code(include_prefix=False)
                if code != '@runtime\n':
                    debug.warning('decorator not found: %s on %s', dec, node)
                return initial
            values = dec_values.execute(arguments.ValuesArguments([values]))
            if not len(values):
                debug.warning('not possible to resolve wrappers found %s', node)
                return initial
        debug.dbg('decorator end %s', values, color='MAGENTA')
    if values != initial:
        return ValueSet([Decoratee(c, decoratee_value) for c in values])
    return values