import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, default, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
from pygments.lexers.data import JsonLexer
class UnixConfigLexer(RegexLexer):
    """
    Lexer for Unix/Linux config files using colon-separated values, e.g.

    * ``/etc/group``
    * ``/etc/passwd``
    * ``/etc/shadow``

    .. versionadded:: 2.12
    """
    name = 'Unix/Linux config files'
    aliases = ['unixconfig', 'linuxconfig']
    filenames = []
    tokens = {'root': [('^#.*', Comment), ('\\n', Whitespace), (':', Punctuation), ('[0-9]+', Number), ('((?!\\n)[a-zA-Z0-9\\_\\-\\s\\(\\),]){2,}', Text), ('[^:\\n]+', String)]}