from pygments.lexer import RegexLexer, bygroups
from pygments.lexer import words as words_
from pygments.lexers._usd_builtins import COMMON_ATTRIBUTES, KEYWORDS, \
from pygments.token import Comment, Keyword, Name, Number, Operator, \
def _keywords(words, type_):
    return [(words_(words, prefix='\\b', suffix='\\b'), type_)]