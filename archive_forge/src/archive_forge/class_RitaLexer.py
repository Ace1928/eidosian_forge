from pygments.lexer import RegexLexer
from pygments.token import Comment, Operator, Keyword, Name, Literal, \
class RitaLexer(RegexLexer):
    """
    Lexer for RITA.

    .. versionadded:: 2.11
    """
    name = 'Rita'
    url = 'https://github.com/zaibacu/rita-dsl'
    filenames = ['*.rita']
    aliases = ['rita']
    mimetypes = ['text/rita']
    tokens = {'root': [('\\n', Whitespace), ('\\s+', Whitespace), ('#(.*?)\\n', Comment.Single), ('@(.*?)\\n', Operator), ('"(\\w|\\d|\\s|(\\\\")|[\\\'_\\-./,\\?\\!])+?"', Literal), ('\\\'(\\w|\\d|\\s|(\\\\\\\')|["_\\-./,\\?\\!])+?\\\'', Literal), ('([A-Z_]+)', Keyword), ('([a-z0-9_]+)', Name), ('((->)|[!?+*|=])', Operator), ('[\\(\\),\\{\\}]', Punctuation)]}