import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches

    Lexer for Fish shell scripts.

    .. versionadded:: 2.1
    