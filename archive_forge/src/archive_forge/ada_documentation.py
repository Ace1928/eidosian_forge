import re
from pygments.lexer import RegexLexer, include, bygroups, words, using, this, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers._ada_builtins import KEYWORD_LIST, BUILTIN_LIST

    For Ada source code.

    .. versionadded:: 1.3
    