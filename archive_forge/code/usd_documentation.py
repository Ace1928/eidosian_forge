from pygments.lexer import RegexLexer, bygroups
from pygments.lexer import words as words_
from pygments.lexers._usd_builtins import COMMON_ATTRIBUTES, KEYWORDS, \
from pygments.token import Comment, Keyword, Name, Number, Operator, \

    A lexer that parses Pixar's Universal Scene Description file format.

    .. versionadded:: 2.6
    