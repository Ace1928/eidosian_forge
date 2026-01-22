import re
from pygments.lexer import RegexLexer
from pygments.token import Punctuation, Text, Comment, Operator, Keyword, \
Lexer for Microsoft Visual FoxPro language.

    FoxPro syntax allows to shorten all keywords and function names
    to 4 characters.  Shortened forms are not recognized by this lexer.

    .. versionadded:: 1.6
    