import re
from pygments.lexer import RegexLexer, include, bygroups, default
from pygments.token import Punctuation, Text, Comment, Operator, Keyword, \

    Lexer for `Limbo programming language <http://www.vitanuova.com/inferno/limbo.html>`_

    TODO:
        - maybe implement better var declaration highlighting
        - some simple syntax error highlighting

    .. versionadded:: 2.0
    