import re
from jedi.inference.names import AbstractArbitraryName
from jedi.inference.helpers import infer_call_of_leaf
from jedi.api.classes import Completion
from jedi.parser_utils import cut_value_at_position
def _matches_quote_at_position(code_lines, quote, position):
    string = code_lines[position[0] - 1][position[1]:position[1] + len(quote)]
    return string == quote