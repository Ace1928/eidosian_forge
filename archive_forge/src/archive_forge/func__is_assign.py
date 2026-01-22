import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _is_assign(self, value):
    if value.endswith('='):
        value = value[:-1].strip()
    var = VariableSplitter(value, identifiers='$@&')
    return var.start == 0 and var.end == len(value)