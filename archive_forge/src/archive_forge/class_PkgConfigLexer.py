import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class PkgConfigLexer(RegexLexer):
    """
    Lexer for `pkg-config
    <http://www.freedesktop.org/wiki/Software/pkg-config/>`_
    (see also `manual page <http://linux.die.net/man/1/pkg-config>`_).

    .. versionadded:: 2.1
    """
    name = 'PkgConfig'
    aliases = ['pkgconfig']
    filenames = ['*.pc']
    mimetypes = []
    tokens = {'root': [('#.*$', Comment.Single), ('^(\\w+)(=)', bygroups(Name.Attribute, Operator)), ('^([\\w.]+)(:)', bygroups(Name.Tag, Punctuation), 'spvalue'), include('interp'), ('[^${}#=:\\n.]+', Text), ('.', Text)], 'interp': [('\\$\\$', Text), ('\\$\\{', String.Interpol, 'curly')], 'curly': [('\\}', String.Interpol, '#pop'), ('\\w+', Name.Attribute)], 'spvalue': [include('interp'), ('#.*$', Comment.Single, '#pop'), ('\\n', Text, '#pop'), ('[^${}#\\n]+', Text), ('.', Text)]}