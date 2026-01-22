import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _waiting_list_variable_index_state(self, char, index):
    if char != '[':
        raise StopIteration
    self._list_and_dict_variable_index_chars.append(char)
    self._state = self._list_variable_index_state