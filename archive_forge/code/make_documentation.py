import re
from pygments.lexer import Lexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer

    Lexer for `CMake <http://cmake.org/Wiki/CMake>`_ files.

    .. versionadded:: 1.2
    