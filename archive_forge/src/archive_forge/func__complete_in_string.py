import re
from textwrap import dedent
from inspect import Parameter
from parso.python.token import PythonTokenTypes
from parso.python import tree
from parso.tree import search_ancestor, Leaf
from parso import split_lines
from jedi import debug
from jedi import settings
from jedi.api import classes
from jedi.api import helpers
from jedi.api import keywords
from jedi.api.strings import complete_dict
from jedi.api.file_name import complete_file_name
from jedi.inference import imports
from jedi.inference.base_value import ValueSet
from jedi.inference.helpers import infer_call_of_leaf, parse_dotted_names
from jedi.inference.context import get_global_filters
from jedi.inference.value import TreeInstance
from jedi.inference.docstring_utils import DocstringModule
from jedi.inference.names import ParamNameWrapper, SubModuleName
from jedi.inference.gradual.conversion import convert_values, convert_names
from jedi.parser_utils import cut_value_at_position
from jedi.plugins import plugin_manager
def _complete_in_string(self, start_leaf, string):
    """
        To make it possible for people to have completions in doctests or
        generally in "Python" code in docstrings, we use the following
        heuristic:

        - Having an indented block of code
        - Having some doctest code that starts with `>>>`
        - Having backticks that doesn't have whitespace inside it
        """

    def iter_relevant_lines(lines):
        include_next_line = False
        for l in code_lines:
            if include_next_line or l.startswith('>>>') or l.startswith(' '):
                yield re.sub('^( *>>> ?| +)', '', l)
            else:
                yield None
            include_next_line = bool(re.match(' *>>>', l))
    string = dedent(string)
    code_lines = split_lines(string, keepends=True)
    relevant_code_lines = list(iter_relevant_lines(code_lines))
    if relevant_code_lines[-1] is not None:
        relevant_code_lines = ['\n' if c is None else c for c in relevant_code_lines]
        return self._complete_code_lines(relevant_code_lines)
    match = re.search('`([^`\\s]+)', code_lines[-1])
    if match:
        return self._complete_code_lines([match.group(1)])
    return []