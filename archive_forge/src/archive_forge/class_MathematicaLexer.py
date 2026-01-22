import re
from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class MathematicaLexer(RegexLexer):
    """
    Lexer for `Mathematica <http://www.wolfram.com/mathematica/>`_ source code.

    .. versionadded:: 2.0
    """
    name = 'Mathematica'
    aliases = ['mathematica', 'mma', 'nb']
    filenames = ['*.nb', '*.cdf', '*.nbp', '*.ma']
    mimetypes = ['application/mathematica', 'application/vnd.wolfram.mathematica', 'application/vnd.wolfram.mathematica.package', 'application/vnd.wolfram.cdf']
    operators = (';;', '=', '=.', '!===', ':=', '->', ':>', '/.', '+', '-', '*', '/', '^', '&&', '||', '!', '<>', '|', '/;', '?', '@', '//', '/@', '@@', '@@@', '~~', '===', '&', '<', '>', '<=', '>=')
    punctuation = (',', ';', '(', ')', '[', ']', '{', '}')

    def _multi_escape(entries):
        return '(%s)' % '|'.join((re.escape(entry) for entry in entries))
    tokens = {'root': [('(?s)\\(\\*.*?\\*\\)', Comment), ('([a-zA-Z]+[A-Za-z0-9]*`)', Name.Namespace), ('([A-Za-z0-9]*_+[A-Za-z0-9]*)', Name.Variable), ('#\\d*', Name.Variable), ('([a-zA-Z]+[a-zA-Z0-9]*)', Name), ('-?\\d+\\.\\d*', Number.Float), ('-?\\d*\\.\\d+', Number.Float), ('-?\\d+', Number.Integer), (words(operators), Operator), (words(punctuation), Punctuation), ('".*?"', String), ('\\s+', Text.Whitespace)]}