from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _suite_nodes_to_string(nodes, pos):
    n = nodes[0]
    prefix, part_of_code = _split_prefix_at(n.get_first_leaf(), pos[0] - 1)
    code = part_of_code + n.get_code(include_prefix=False) + ''.join((n.get_code() for n in nodes[1:]))
    return (prefix, code)