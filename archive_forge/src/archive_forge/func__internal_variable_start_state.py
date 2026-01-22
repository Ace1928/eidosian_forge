import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _internal_variable_start_state(self, char, index):
    self._state = self._variable_state
    if char == '{':
        self._variable_chars.append(char)
        self._open_curly += 1
        self._may_have_internal_variables = True
    else:
        self._variable_state(char, index)