from pygments.lexer import RegexLexer, bygroups, default, include
from pygments.token import Text, Operator, Keyword, Punctuation, Name, \
class ArrowLexer(RegexLexer):
    """
    Lexer for Arrow

    .. versionadded:: 2.7
    """
    name = 'Arrow'
    url = 'https://pypi.org/project/py-arrow-lang/'
    aliases = ['arrow']
    filenames = ['*.arw']
    tokens = {'root': [('\\s+', Whitespace), ('^[|\\s]+', Punctuation), include('blocks'), include('statements'), include('expressions')], 'blocks': [('(function)(\\n+)(/-->)(\\s*)' + DECL + '(\\()', bygroups(Keyword.Reserved, Whitespace, Punctuation, Whitespace, Keyword.Type, Punctuation, Whitespace, Name.Function, Punctuation), 'fparams'), ('/-->$|\\\\-->$|/--<|\\\\--<|\\^', Punctuation)], 'statements': [(DECL, bygroups(Keyword.Type, Punctuation, Text, Name.Variable)), ('\\[', Punctuation, 'index'), ('=', Operator), ('require|main', Keyword.Reserved), ('print', Keyword.Reserved, 'print')], 'expressions': [('\\s+', Whitespace), ('[0-9]+', Number.Integer), ('true|false', Keyword.Constant), ("'", String.Char, 'char'), ('"', String.Double, 'string'), ('\\{', Punctuation, 'array'), ('==|!=|<|>|\\+|-|\\*|/|%', Operator), ('and|or|not|length', Operator.Word), ('(input)(\\s+)(int|char\\[\\])', bygroups(Keyword.Reserved, Whitespace, Keyword.Type)), (IDENT + '(\\()', bygroups(Name.Function, Punctuation), 'fargs'), (IDENT, Name.Variable), ('\\[', Punctuation, 'index'), ('\\(', Punctuation, 'expressions'), ('\\)', Punctuation, '#pop')], 'print': [include('expressions'), (',', Punctuation), default('#pop')], 'fparams': [(DECL, bygroups(Keyword.Type, Punctuation, Whitespace, Name.Variable)), (',', Punctuation), ('\\)', Punctuation, '#pop')], 'escape': [('\\\\(["\\\\/abfnrtv]|[0-9]{1,3}|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4})', String.Escape)], 'char': [("'", String.Char, '#pop'), include('escape'), ("[^'\\\\]", String.Char)], 'string': [('"', String.Double, '#pop'), include('escape'), ('[^"\\\\]+', String.Double)], 'array': [include('expressions'), ('\\}', Punctuation, '#pop'), (',', Punctuation)], 'fargs': [include('expressions'), ('\\)', Punctuation, '#pop'), (',', Punctuation)], 'index': [include('expressions'), ('\\]', Punctuation, '#pop')]}