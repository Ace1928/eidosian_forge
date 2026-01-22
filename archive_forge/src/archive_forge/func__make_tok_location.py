import re
from .ply import lex
from .ply.lex import TOKEN
def _make_tok_location(self, token):
    return (token.lineno, self.find_tok_column(token))