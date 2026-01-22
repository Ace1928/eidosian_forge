import re
from pygments.lexer import RegexLexer, include, bygroups, words, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.python import PythonLexer
def _process_declarations(self, tokens):
    opening_paren = False
    for index, token, value in tokens:
        yield (index, token, value)
        if self._relevant(token):
            if opening_paren and token == Keyword and (value in self.DECLARATIONS):
                declaration = value
                for index, token, value in self._process_declaration(declaration, tokens):
                    yield (index, token, value)
            opening_paren = value == '(' and token == Punctuation