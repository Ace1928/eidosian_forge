import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, using, \
from pygments.util import get_bool_opt, shebang_matches
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni

    A Python lexer recognizing Numerical Python builtins.

    .. versionadded:: 0.10
    