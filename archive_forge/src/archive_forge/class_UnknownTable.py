import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class UnknownTable(_Table):
    _tokenizer_class = Comment

    def _continues(self, value, index):
        return False