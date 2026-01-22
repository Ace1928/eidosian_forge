from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _find_needed_output_variables(context, search_node, at_least_pos, return_variables):
    """
    Searches everything after at_least_pos in a node and checks if any of the
    return_variables are used in there and returns those.
    """
    for node in search_node.children:
        if node.start_pos < at_least_pos:
            continue
        return_variables = set(return_variables)
        for name in _find_non_global_names([node]):
            if not name.is_definition() and name.value in return_variables:
                return_variables.remove(name.value)
                yield name.value