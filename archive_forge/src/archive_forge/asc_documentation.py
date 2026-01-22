import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Comment, Generic, Name, Operator, String, Whitespace

    Lexer for ASCII armored files, containing `-----BEGIN/END ...-----` wrapped
    base64 data.

    .. versionadded:: 2.10
    