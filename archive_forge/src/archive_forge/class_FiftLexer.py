from pygments.lexer import RegexLexer, include
from pygments.token import Literal, Comment, Name, String, Number, Whitespace
class FiftLexer(RegexLexer):
    """
    For Fift source code.
    """
    name = 'Fift'
    aliases = ['fift', 'fif']
    filenames = ['*.fif']
    url = 'https://ton-blockchain.github.io/docs/fiftbase.pdf'
    tokens = {'root': [('\\s+', Whitespace), include('comments'), ('[\\.+]?\\"', String, 'string'), ('0x[0-9a-fA-F]+', Number.Hex), ('0b[01]+', Number.Bin), ('-?[0-9]+("/"-?[0-9]+)?', Number.Decimal), ('b\\{[01]+\\}', Literal), ('x\\{[0-9a-fA-F_]+\\}', Literal), ('B\\{[0-9a-fA-F_]+\\}', Literal), ('\\S+', Name)], 'string': [('\\\\.', String.Escape), ('\\"', String, '#pop'), ('[^\\"\\r\\n\\\\]+', String)], 'comments': [('//.*', Comment.Singleline), ('/\\*', Comment.Multiline, 'comment')], 'comment': [('[^/*]+', Comment.Multiline), ('/\\*', Comment.Multiline, '#push'), ('\\*/', Comment.Multiline, '#pop'), ('[*/]', Comment.Multiline)]}