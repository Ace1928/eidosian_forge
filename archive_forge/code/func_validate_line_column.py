import re
from collections import namedtuple
from textwrap import dedent
from itertools import chain
from functools import wraps
from inspect import Parameter
from parso.python.parser import Parser
from parso.python import tree
from jedi.inference.base_value import NO_VALUES
from jedi.inference.syntax_tree import infer_atom
from jedi.inference.helpers import infer_call_of_leaf
from jedi.inference.compiled import get_string_value_set
from jedi.cache import signature_time_cache, memoize_method
from jedi.parser_utils import get_parent_scope
def validate_line_column(func):

    @wraps(func)
    def wrapper(self, line=None, column=None, *args, **kwargs):
        line = max(len(self._code_lines), 1) if line is None else line
        if not 0 < line <= len(self._code_lines):
            raise ValueError('`line` parameter is not in a valid range.')
        line_string = self._code_lines[line - 1]
        line_len = len(line_string)
        if line_string.endswith('\r\n'):
            line_len -= 2
        elif line_string.endswith('\n'):
            line_len -= 1
        column = line_len if column is None else column
        if not 0 <= column <= line_len:
            raise ValueError('`column` parameter (%d) is not in a valid range (0-%d) for line %d (%r).' % (column, line_len, line, line_string))
        return func(self, line, column, *args, **kwargs)
    return wrapper