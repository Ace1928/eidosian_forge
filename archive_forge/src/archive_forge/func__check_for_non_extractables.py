from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _check_for_non_extractables(nodes):
    for n in nodes:
        try:
            children = n.children
        except AttributeError:
            if n.value == 'return':
                raise RefactoringError('Can only extract return statements if they are at the end.')
            if n.value == 'yield':
                raise RefactoringError('Cannot extract yield statements.')
        else:
            _check_for_non_extractables(children)