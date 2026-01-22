from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _is_name_input(module_context, names, first, last):
    for name in names:
        if name.api_type == 'param' or not name.parent_context.is_module():
            if name.get_root_context() is not module_context:
                return True
            if name.start_pos is None or not first <= name.start_pos < last:
                return True
    return False