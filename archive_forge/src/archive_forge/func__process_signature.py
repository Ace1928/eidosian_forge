import re
from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
def _process_signature(self, tokens):
    for index, token, value in tokens:
        if token == Literal and value == '}':
            yield (index, Punctuation, value)
            raise StopIteration
        elif token in (Literal, Name.Function):
            token = Name.Variable if value.istitle() else Keyword.Type
        yield (index, token, value)