import re
from pygments.lexer import RegexLexer, bygroups, default
from pygments.token import Keyword, Punctuation, String, Number, Operator, Generic, \
class TurtleLexer(RegexLexer):
    """
    Lexer for `Turtle <http://www.w3.org/TR/turtle/>`_ data language.

    .. versionadded:: 2.1
    """
    name = 'Turtle'
    aliases = ['turtle']
    filenames = ['*.ttl']
    mimetypes = ['text/turtle', 'application/x-turtle']
    flags = re.IGNORECASE
    patterns = {'PNAME_NS': '((?:[a-z][\\w-]*)?\\:)', 'IRIREF': '(<[^<>"{}|^`\\\\\\x00-\\x20]*>)'}
    patterns['PrefixedName'] = '%(PNAME_NS)s([a-z][\\w-]*)' % patterns
    tokens = {'root': [('\\s+', Whitespace), ('(@base|BASE)(\\s+)%(IRIREF)s(\\s*)(\\.?)' % patterns, bygroups(Keyword, Whitespace, Name.Variable, Whitespace, Punctuation)), ('(@prefix|PREFIX)(\\s+)%(PNAME_NS)s(\\s+)%(IRIREF)s(\\s*)(\\.?)' % patterns, bygroups(Keyword, Whitespace, Name.Namespace, Whitespace, Name.Variable, Whitespace, Punctuation)), ('(?<=\\s)a(?=\\s)', Keyword.Type), ('%(IRIREF)s' % patterns, Name.Variable), ('%(PrefixedName)s' % patterns, bygroups(Name.Namespace, Name.Tag)), ('#[^\\n]+', Comment), ('\\b(true|false)\\b', Literal), ('[+\\-]?\\d*\\.\\d+', Number.Float), ('[+\\-]?\\d*(:?\\.\\d+)?E[+\\-]?\\d+', Number.Float), ('[+\\-]?\\d+', Number.Integer), ('[\\[\\](){}.;,:^]', Punctuation), ('"""', String, 'triple-double-quoted-string'), ('"', String, 'single-double-quoted-string'), ("'''", String, 'triple-single-quoted-string'), ("'", String, 'single-single-quoted-string')], 'triple-double-quoted-string': [('"""', String, 'end-of-string'), ('[^\\\\]+', String), ('\\\\', String, 'string-escape')], 'single-double-quoted-string': [('"', String, 'end-of-string'), ('[^"\\\\\\n]+', String), ('\\\\', String, 'string-escape')], 'triple-single-quoted-string': [("'''", String, 'end-of-string'), ('[^\\\\]+', String), ('\\\\', String, 'string-escape')], 'single-single-quoted-string': [("'", String, 'end-of-string'), ("[^'\\\\\\n]+", String), ('\\\\', String, 'string-escape')], 'string-escape': [('.', String, '#pop')], 'end-of-string': [('(@)([a-z]+(:?-[a-z0-9]+)*)', bygroups(Operator, Generic.Emph), '#pop:2'), ('(\\^\\^)%(IRIREF)s' % patterns, bygroups(Operator, Generic.Emph), '#pop:2'), ('(\\^\\^)%(PrefixedName)s' % patterns, bygroups(Operator, Generic.Emph, Generic.Emph), '#pop:2'), default('#pop:2')]}