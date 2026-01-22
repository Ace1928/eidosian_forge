import re
from pygments.token import Comment, Operator, Keyword, Name, String, \
from pygments.lexer import RegexLexer, bygroups, include
class DnsZoneLexer(RegexLexer):
    """
    Lexer for DNS zone file

    .. versionadded:: 2.16
    """
    flags = re.MULTILINE
    name = 'Zone'
    aliases = ['zone']
    filenames = ['*.zone']
    url = 'https://datatracker.ietf.org/doc/html/rfc1035'
    mimetypes = ['text/dns']
    tokens = {'root': [('([ \\t]*)(;.*)(\\n)', bygroups(Whitespace, Comment.Single, Whitespace)), ('^\\$ORIGIN\\b', Keyword, 'values'), ('^\\$TTL\\b', Keyword, 'values'), ('^\\$INCLUDE\\b', Comment.Preproc, 'include'), ('^\\$[A-Z]+\\b', Keyword, 'values'), ('^(@)([ \\t]+)(?:([0-9]+[smhdw]?)([ \\t]+))?(?:' + CLASSES_RE + '([ \t]+))?([A-Z]+)([ \t]+)', bygroups(Operator, Whitespace, Number.Integer, Whitespace, Name.Class, Whitespace, Keyword.Type, Whitespace), 'values'), ('^([^ \\t\\n]*)([ \\t]+)(?:([0-9]+[smhdw]?)([ \\t]+))?(?:' + CLASSES_RE + '([ \t]+))?([A-Z]+)([ \t]+)', bygroups(Name, Whitespace, Number.Integer, Whitespace, Name.Class, Whitespace, Keyword.Type, Whitespace), 'values'), ('^(Operator)([ \\t]+)(?:' + CLASSES_RE + '([ \t]+))?(?:([0-9]+[smhdw]?)([ \t]+))?([A-Z]+)([ \t]+)', bygroups(Name, Whitespace, Number.Integer, Whitespace, Name.Class, Whitespace, Keyword.Type, Whitespace), 'values'), ('^([^ \\t\\n]*)([ \\t]+)(?:' + CLASSES_RE + '([ \t]+))?(?:([0-9]+[smhdw]?)([ \t]+))?([A-Z]+)([ \t]+)', bygroups(Name, Whitespace, Number.Integer, Whitespace, Name.Class, Whitespace, Keyword.Type, Whitespace), 'values')], 'values': [('\\n', Whitespace, '#pop'), ('\\(', Punctuation, 'nested'), include('simple-values')], 'nested': [('\\)', Punctuation, '#pop'), include('simple-values')], 'simple-values': [('(;.*)(\\n)', bygroups(Comment.Single, Whitespace)), ('[ \\t]+', Whitespace), ('@\\b', Operator), ('"', String, 'string'), ('[0-9]+[smhdw]?$', Number.Integer), ('([0-9]+[smhdw]?)([ \\t]+)', bygroups(Number.Integer, Whitespace)), ('\\S+', Literal)], 'include': [('([ \\t]+)([^ \\t\\n]+)([ \\t]+)([-\\._a-zA-Z]+)([ \\t]+)(;.*)?$', bygroups(Whitespace, Comment.PreprocFile, Whitespace, Name, Whitespace, Comment.Single), '#pop'), ('([ \\t]+)([^ \\t\\n]+)([ \\t\\n]+)$', bygroups(Whitespace, Comment.PreprocFile, Whitespace), '#pop')], 'string': [('\\\\"', String), ('"', String, '#pop'), ('[^"]+', String)]}

    def analyse_text(text):
        return text.startswith('$ORIGIN')