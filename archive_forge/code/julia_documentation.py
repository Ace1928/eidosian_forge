import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches, unirange

    For Julia console sessions. Modeled after MatlabSessionLexer.

    .. versionadded:: 1.6
    