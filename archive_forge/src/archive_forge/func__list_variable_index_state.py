import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _list_variable_index_state(self, char, index):
    self._list_and_dict_variable_index_chars.append(char)
    if char == ']':
        raise StopIteration