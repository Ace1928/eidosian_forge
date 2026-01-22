import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class PacmanConfLexer(RegexLexer):
    """
    Lexer for `pacman.conf
    <https://www.archlinux.org/pacman/pacman.conf.5.html>`_.

    Actually, IniLexer works almost fine for this format,
    but it yield error token. It is because pacman.conf has
    a form without assignment like:

        UseSyslog
        Color
        TotalDownload
        CheckSpace
        VerbosePkgLists

    These are flags to switch on.

    .. versionadded:: 2.1
    """
    name = 'PacmanConf'
    aliases = ['pacmanconf']
    filenames = ['pacman.conf']
    mimetypes = []
    tokens = {'root': [('#.*$', Comment.Single), ('^\\s*\\[.*?\\]\\s*$', Keyword), ('(\\w+)(\\s*)(=)', bygroups(Name.Attribute, Text, Operator)), ('^(\\s*)(\\w+)(\\s*)$', bygroups(Text, Name.Attribute, Text)), (words(('$repo', '$arch', '%o', '%u'), suffix='\\b'), Name.Variable), ('.', Text)]}