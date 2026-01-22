import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, default, words, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
from pygments.lexers.data import JsonLexer
class SystemdLexer(RegexLexer):
    """
    Lexer for systemd unit files.

    .. versionadded:: 2.16
    """
    name = 'Systemd'
    url = 'https://www.freedesktop.org/software/systemd/man/systemd.syntax.html'
    aliases = ['systemd']
    filenames = ['*.service', '*.socket', '*.device', '*.mount', '*.automount', '*.swap', '*.target', '*.path', '*.timer', '*.slice', '*.scope']
    tokens = {'root': [('^[ \\t]*\\n', Whitespace), ('^([;#].*)(\\n)', bygroups(Comment.Single, Whitespace)), ('(\\[[^\\]\\n]+\\])(\\n)', bygroups(Keyword, Whitespace)), ('([^=]+)([ \\t]*)(=)([ \\t]*)([^\\n]*)(\\\\)(\\n)', bygroups(Name.Attribute, Whitespace, Operator, Whitespace, String, Text, Whitespace), 'value'), ('([^=]+)([ \\t]*)(=)([ \\t]*)([^\\n]*)(\\n)', bygroups(Name.Attribute, Whitespace, Operator, Whitespace, String, Whitespace))], 'value': [('^([;#].*)(\\n)', bygroups(Comment.Single, Whitespace)), ('([ \\t]*)([^\\n]*)(\\\\)(\\n)', bygroups(Whitespace, String, Text, Whitespace)), ('([ \\t]*)([^\\n]*)(\\n)', bygroups(Whitespace, String, Whitespace), '#pop')]}

    def analyse_text(text):
        if text.startswith('[Unit]'):
            return 1.0
        if re.search('^\\[Unit\\][ \\t]*$', text[:500], re.MULTILINE) is not None:
            return 0.9
        return 0.0