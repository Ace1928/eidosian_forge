from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _get_code_insertion_node(node, is_bound_method):
    if not is_bound_method or function_is_staticmethod(node):
        while node.parent.type != 'file_input':
            node = node.parent
    while node.parent.type in ('async_funcdef', 'decorated', 'async_stmt'):
        node = node.parent
    return node