import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, words, include
from pygments.token import Comment, Text, Keyword, String, Number, Literal, \
from pygments.lexers.web import HtmlLexer, XmlLexer, CssLexer, JavascriptLexer
from pygments.lexers.python import PythonLexer
class CSSUL4Lexer(DelegatingLexer):
    """
    Lexer for UL4 embedded in CSS.
    """
    name = 'CSS+UL4'
    aliases = ['css+ul4']
    filenames = ['*.cssul4']

    def __init__(self, **options):
        super().__init__(CssLexer, UL4Lexer, **options)