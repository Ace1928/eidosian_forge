import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
class MoinWikiLexer(RegexLexer):
    """
    For MoinMoin (and Trac) Wiki markup.

    .. versionadded:: 0.7
    """
    name = 'MoinMoin/Trac Wiki markup'
    aliases = ['trac-wiki', 'moin']
    filenames = []
    mimetypes = ['text/x-trac-wiki']
    flags = re.MULTILINE | re.IGNORECASE
    tokens = {'root': [('^#.*$', Comment), ('(!)(\\S+)', bygroups(Keyword, Text)), ('^(=+)([^=]+)(=+)(\\s*#.+)?$', bygroups(Generic.Heading, using(this), Generic.Heading, String)), ('(\\{\\{\\{)(\\n#!.+)?', bygroups(Name.Builtin, Name.Namespace), 'codeblock'), ("(\\'\\'\\'?|\\|\\||`|__|~~|\\^|,,|::)", Comment), ('^( +)([.*-])( )', bygroups(Text, Name.Builtin, Text)), ('^( +)([a-z]{1,5}\\.)( )', bygroups(Text, Name.Builtin, Text)), ('\\[\\[\\w+.*?\\]\\]', Keyword), ('(\\[[^\\s\\]]+)(\\s+[^\\]]+?)?(\\])', bygroups(Keyword, String, Keyword)), ('^----+$', Keyword), ("[^\\n\\'\\[{!_~^,|]+", Text), ('\\n', Text), ('.', Text)], 'codeblock': [('\\}\\}\\}', Name.Builtin, '#pop'), ('\\{\\{\\{', Text, '#push'), ('[^{}]+', Comment.Preproc), ('.', Comment.Preproc)]}