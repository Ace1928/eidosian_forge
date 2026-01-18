from pygments.lexer import RegexLexer, bygroups
from pygments.token import Name, Number, String, Text, Punctuation

    Lexer for Procfile file format.

    The format is used to run processes on Heroku or is used by Foreman or
    Honcho tools.

    .. versionadded:: 2.10
    