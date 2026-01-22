import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _find_variable(self, string):
    max_end_index = string.rfind('}')
    if max_end_index == -1:
        raise ValueError('No variable end found')
    if self._is_escaped(string, max_end_index):
        return self._find_variable(string[:max_end_index])
    start_index = self._find_start_index(string, 1, max_end_index)
    if start_index == -1:
        raise ValueError('No variable start found')
    return (start_index, max_end_index)