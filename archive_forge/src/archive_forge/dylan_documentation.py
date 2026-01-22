import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \

    For Dylan interactive console output like:

    .. sourcecode:: dylan-console

        ? let a = 1;
        => 1
        ? a
        => 1

    This is based on a copy of the RubyConsoleLexer.

    .. versionadded:: 1.6
    