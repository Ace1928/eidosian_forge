from pygments.lexer import RegexLexer, words
from pygments.token import Comment, String, Name, Text, Punctuation, \
class YaraLexer(RegexLexer):
    """
    For YARA rules

    .. versionadded:: 2.16
    """
    name = 'YARA'
    url = 'https://virustotal.github.io/yara/'
    aliases = ['yara', 'yar']
    filenames = ['*.yar']
    mimetypes = ['text/x-yara']
    tokens = {'root': [('\\s+', Whitespace), ('//.*?$', Comment.Single), ('\\#.*?$', Comment.Single), ('/\\*', Comment.Multiline, 'comment'), (words(('rule', 'private', 'global', 'import', 'include'), prefix='\\b', suffix='\\b'), Keyword.Declaration), (words(('strings', 'condition', 'meta'), prefix='\\b', suffix='\\b'), Keyword), (words(('ascii', 'at', 'base64', 'base64wide', 'condition', 'contains', 'endswith', 'entrypoint', 'filesize', 'for', 'fullword', 'icontains', 'iendswith', 'iequals', 'in', 'include', 'int16', 'int16be', 'int32', 'int32be', 'int8', 'int8be', 'istartswith', 'matches', 'meta', 'nocase', 'none', 'of', 'startswith', 'strings', 'them', 'uint16', 'uint16be', 'uint32', 'uint32be', 'uint8', 'uint8be', 'wide', 'xor', 'defined'), prefix='\\b', suffix='\\b'), Name.Builtin), ('(true|false)\\b', Keyword.Constant), ('(and|or|not|any|all)\\b', Operator.Word), ('(\\$\\w+)', Name.Variable), ('"[^"]*"', String.Double), ("\\'[^\\']*\\'", String.Single), ('\\{.*?\\}$', Number.Hex), ('(/.*?/)', String.Regex), ('[a-z_]\\w*', Name), ('[$(){}[\\].?+*|]', Punctuation), ('[:=,;]', Punctuation), ('.', Text)], 'comment': [('[^*/]+', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)]}