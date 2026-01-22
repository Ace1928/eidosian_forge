import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
def _is_template_set(self, template):
    return normalize(template) not in ('', '\\', 'none', '${empty}')