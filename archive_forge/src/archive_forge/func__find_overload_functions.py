from parso.python import tree
from jedi import debug
from jedi.inference.cache import inference_state_method_cache, CachedMetaClass
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import docstrings
from jedi.inference import flow_analysis
from jedi.inference.signature import TreeSignature
from jedi.inference.filters import ParserTreeFilter, FunctionExecutionFilter, \
from jedi.inference.names import ValueName, AbstractNameDefinition, \
from jedi.inference.base_value import ContextualizedNode, NO_VALUES, \
from jedi.inference.lazy_value import LazyKnownValues, LazyKnownValue, \
from jedi.inference.context import ValueContext, TreeContextMixin
from jedi.inference.value import iterable
from jedi import parser_utils
from jedi.inference.parser_cache import get_yield_exprs
from jedi.inference.helpers import values_from_qualified_names
from jedi.inference.gradual.generics import TupleGenericManager
def _find_overload_functions(context, tree_node):

    def _is_overload_decorated(funcdef):
        if funcdef.parent.type == 'decorated':
            decorators = funcdef.parent.children[0]
            if decorators.type == 'decorator':
                decorators = [decorators]
            else:
                decorators = decorators.children
            for decorator in decorators:
                dotted_name = decorator.children[1]
                if dotted_name.type == 'name' and dotted_name.value == 'overload':
                    return True
        return False
    if tree_node.type == 'lambdef':
        return
    if _is_overload_decorated(tree_node):
        yield tree_node
    while True:
        filter = ParserTreeFilter(context, until_position=tree_node.start_pos)
        names = filter.get(tree_node.name.value)
        assert isinstance(names, list)
        if not names:
            break
        found = False
        for name in names:
            funcdef = name.tree_name.parent
            if funcdef.type == 'funcdef' and _is_overload_decorated(funcdef):
                tree_node = funcdef
                found = True
                yield funcdef
        if not found:
            break