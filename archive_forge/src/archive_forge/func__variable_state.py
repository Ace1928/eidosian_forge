import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _variable_state(self, char, index):
    self._variable_chars.append(char)
    if char == '}' and (not self._is_escaped(self._string, index)):
        self._open_curly -= 1
        if self._open_curly == 0:
            if not self._is_list_or_dict_variable():
                raise StopIteration
            self._state = self._waiting_list_variable_index_state
    elif char in self._identifiers:
        self._state = self._internal_variable_start_state