import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, words, include
from pygments.token import Comment, Text, Keyword, String, Number, Literal, \
from pygments.lexers.web import HtmlLexer, XmlLexer, CssLexer, JavascriptLexer
from pygments.lexers.python import PythonLexer
class XMLUL4Lexer(DelegatingLexer):
    """
    Lexer for UL4 embedded in XML.
    """
    name = 'XML+UL4'
    aliases = ['xml+ul4']
    filenames = ['*.xmlul4']

    def __init__(self, **options):
        super().__init__(XmlLexer, UL4Lexer, **options)