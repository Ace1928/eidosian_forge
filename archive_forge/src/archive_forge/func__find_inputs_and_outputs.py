from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _find_inputs_and_outputs(module_context, context, nodes):
    first = nodes[0].start_pos
    last = nodes[-1].end_pos
    inputs = []
    outputs = []
    for name in _find_non_global_names(nodes):
        if name.is_definition():
            if name not in outputs:
                outputs.append(name.value)
        elif name.value not in inputs:
            name_definitions = context.goto(name, name.start_pos)
            if not name_definitions or _is_name_input(module_context, name_definitions, first, last):
                inputs.append(name.value)
    return (inputs, outputs)