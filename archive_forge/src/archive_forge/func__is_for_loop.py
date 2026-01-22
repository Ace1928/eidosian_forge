import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _is_for_loop(self, value):
    return value.startswith(':') and normalize(value, remove=':') == 'for'