import re
from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class GAPLexer(RegexLexer):
    """
    For `GAP <http://www.gap-system.org>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'GAP'
    aliases = ['gap']
    filenames = ['*.g', '*.gd', '*.gi', '*.gap']
    tokens = {'root': [('#.*$', Comment.Single), ('"(?:[^"\\\\]|\\\\.)*"', String), ('\\(|\\)|\\[|\\]|\\{|\\}', Punctuation), ('(?x)\\b(?:\n                if|then|elif|else|fi|\n                for|while|do|od|\n                repeat|until|\n                break|continue|\n                function|local|return|end|\n                rec|\n                quit|QUIT|\n                IsBound|Unbind|\n                TryNextMethod|\n                Info|Assert\n              )\\b', Keyword), ('(?x)\\b(?:\n                true|false|fail|infinity\n              )\\b', Name.Constant), ('(?x)\\b(?:\n                (Declare|Install)([A-Z][A-Za-z]+)|\n                   BindGlobal|BIND_GLOBAL\n              )\\b', Name.Builtin), ('\\.|,|:=|;|=|\\+|-|\\*|/|\\^|>|<', Operator), ('(?x)\\b(?:\n                and|or|not|mod|in\n              )\\b', Operator.Word), ('(?x)\n              (?:\\w+|`[^`]*`)\n              (?:::\\w+|`[^`]*`)*', Name.Variable), ('[0-9]+(?:\\.[0-9]*)?(?:e[0-9]+)?', Number), ('\\.[0-9]+(?:e[0-9]+)?', Number), ('.', Text)]}