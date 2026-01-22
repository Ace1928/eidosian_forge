import re
from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
def _process_symbols(self, tokens):
    opening_paren = False
    for index, token, value in tokens:
        if opening_paren and token in (Literal, Name.Variable):
            token = self.MAPPINGS.get(value, Name.Function)
        elif token == Literal and value in self.BUILTINS_ANYWHERE:
            token = Name.Builtin
        opening_paren = value == '(' and token == Punctuation
        yield (index, token, value)