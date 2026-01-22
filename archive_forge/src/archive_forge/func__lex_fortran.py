import re
from pygments.lexer import RegexLexer, bygroups, include, words, using, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def _lex_fortran(self, match, ctx=None):
    """Lex a line just as free form fortran without line break."""
    lexer = FortranLexer()
    text = match.group(0) + '\n'
    for index, token, value in lexer.get_tokens_unprocessed(text):
        value = value.replace('\n', '')
        if value != '':
            yield (index, token, value)