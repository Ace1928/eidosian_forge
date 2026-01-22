import re
from pygments.lexer import ExtendedRegexLexer, RegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class HxmlLexer(RegexLexer):
    """
    Lexer for `haXe build <http://haxe.org/doc/compiler>`_ files.

    .. versionadded:: 1.6
    """
    name = 'Hxml'
    aliases = ['haxeml', 'hxml']
    filenames = ['*.hxml']
    tokens = {'root': [('(--)(next)', bygroups(Punctuation, Generic.Heading)), ('(-)(prompt|debug|v)', bygroups(Punctuation, Keyword.Keyword)), ('(--)(neko-source|flash-strict|flash-use-stage|no-opt|no-traces|no-inline|times|no-output)', bygroups(Punctuation, Keyword)), ('(-)(cpp|js|neko|x|as3|swf9?|swf-lib|php|xml|main|lib|D|resource|cp|cmd)( +)(.+)', bygroups(Punctuation, Keyword, Whitespace, String)), ('(-)(swf-version)( +)(\\d+)', bygroups(Punctuation, Keyword, Number.Integer)), ('(-)(swf-header)( +)(\\d+)(:)(\\d+)(:)(\\d+)(:)([A-Fa-f0-9]{6})', bygroups(Punctuation, Keyword, Whitespace, Number.Integer, Punctuation, Number.Integer, Punctuation, Number.Integer, Punctuation, Number.Hex)), ('(--)(js-namespace|php-front|php-lib|remap|gen-hx-classes)( +)(.+)', bygroups(Punctuation, Keyword, Whitespace, String)), ('#.*', Comment.Single)]}