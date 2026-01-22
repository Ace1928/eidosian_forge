import re
from pygments.lexer import RegexLexer, default, words, bygroups, include, using
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.shell import BashLexer
class RegeditLexer(RegexLexer):
    """
    Lexer for `Windows Registry
    <http://en.wikipedia.org/wiki/Windows_Registry#.REG_files>`_ files produced
    by regedit.

    .. versionadded:: 1.6
    """
    name = 'reg'
    aliases = ['registry']
    filenames = ['*.reg']
    mimetypes = ['text/x-windows-registry']
    tokens = {'root': [('Windows Registry Editor.*', Text), ('\\s+', Text), ('[;#].*', Comment.Single), ('(\\[)(-?)(HKEY_[A-Z_]+)(.*?\\])$', bygroups(Keyword, Operator, Name.Builtin, Keyword)), ('("(?:\\\\"|\\\\\\\\|[^"])+")([ \\t]*)(=)([ \\t]*)', bygroups(Name.Attribute, Text, Operator, Text), 'value'), ('(.*?)([ \\t]*)(=)([ \\t]*)', bygroups(Name.Attribute, Text, Operator, Text), 'value')], 'value': [('-', Operator, '#pop'), ('(dword|hex(?:\\([0-9a-fA-F]\\))?)(:)([0-9a-fA-F,]+)', bygroups(Name.Variable, Punctuation, Number), '#pop'), ('.+', String, '#pop'), default('#pop')]}

    def analyse_text(text):
        return text.startswith('Windows Registry Editor')