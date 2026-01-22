import re
import warnings
from parso import parse, ParserSyntaxError
from jedi import debug
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import iterator_to_value_set, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValues
def _infer_for_statement_string(module_context, string):
    if string is None:
        return []
    potential_imports = re.findall('((?:\\w+\\.)*\\w+)\\.', string)
    imports = '\n'.join((f'import {p}' for p in potential_imports))
    string = f'{imports}\n{string}'
    debug.dbg('Parse docstring code %s', string, color='BLUE')
    grammar = module_context.inference_state.grammar
    try:
        module = grammar.parse(string, error_recovery=False)
    except ParserSyntaxError:
        return []
    try:
        stmt = module.children[-2]
    except (AttributeError, IndexError):
        return []
    if stmt.type not in ('name', 'atom', 'atom_expr'):
        return []
    from jedi.inference.docstring_utils import DocstringModule
    m = DocstringModule(in_module_context=module_context, inference_state=module_context.inference_state, module_node=module, code_lines=[])
    return list(_execute_types_in_stmt(m.as_context(), stmt))