import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import looks_like_xml, html_doctype_matches
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.jvm import ScalaLexer
from pygments.lexers.css import CssLexer, _indentation, _starts_block
from pygments.lexers.ruby import RubyLexer
class UrlEncodedLexer(RegexLexer):
    """
    Lexer for urlencoded data

    .. versionadded:: 2.16
    """
    name = 'urlencoded'
    aliases = ['urlencoded']
    mimetypes = ['application/x-www-form-urlencoded']
    tokens = {'root': [('([^&=]*)(=)([^=&]*)(&?)', bygroups(Name.Tag, Operator, String, Punctuation))]}