import re
from pygments.lexer import Lexer, RegexLexer, do_insertions, bygroups, \
from pygments.token import Punctuation, \
from pygments.util import shebang_matches
class BashSessionLexer(ShellSessionBaseLexer):
    """
    Lexer for simplistic shell sessions.

    .. versionadded:: 1.1
    """
    name = 'Bash Session'
    aliases = ['console', 'shell-session']
    filenames = ['*.sh-session', '*.shell-session']
    mimetypes = ['application/x-shell-session', 'application/x-sh-session']
    _innerLexerCls = BashLexer
    _ps1rgx = '^((?:(?:\\[.*?\\])|(?:\\(\\S+\\))?(?:| |sh\\S*?|\\w+\\S+[@:]\\S+(?:\\s+\\S+)?|\\[\\S+[@:][^\\n]+\\].+))\\s*[$#%])(.*\\n?)'
    _ps2 = '>'