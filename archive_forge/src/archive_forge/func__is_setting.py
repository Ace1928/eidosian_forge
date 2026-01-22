import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _is_setting(self, value):
    return value.startswith('[') and value.endswith(']')