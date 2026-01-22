import re
from pygments.lexer import RegexLexer, DelegatingLexer, bygroups, words, include
from pygments.token import Comment, Text, Keyword, String, Number, Literal, \
from pygments.lexers.web import HtmlLexer, XmlLexer, CssLexer, JavascriptLexer
from pygments.lexers.python import PythonLexer
class PythonUL4Lexer(DelegatingLexer):
    """
    Lexer for UL4 embedded in Python.
    """
    name = 'Python+UL4'
    aliases = ['py+ul4']
    filenames = ['*.pyul4']

    def __init__(self, **options):
        super().__init__(PythonLexer, UL4Lexer, **options)