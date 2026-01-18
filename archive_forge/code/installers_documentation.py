import re
from pygments.lexer import RegexLexer, include, bygroups, using, this, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \

    Lexer for Debian ``control`` files and ``apt-cache show <pkg>`` outputs.

    .. versionadded:: 0.9
    