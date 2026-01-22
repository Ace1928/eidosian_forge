import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer, LassoLexer
from pygments.lexers.css import CssLexer
from pygments.lexers.php import PhpLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.perl import PerlLexer
from pygments.lexers.jvm import JavaLexer, TeaLangLexer
from pygments.lexers.data import YamlLexer
from pygments.lexer import Lexer, DelegatingLexer, RegexLexer, bygroups, \
from pygments.token import Error, Punctuation, Whitespace, \
from pygments.util import html_doctype_matches, looks_like_xml
class TwigHtmlLexer(DelegatingLexer):
    """
    Subclass of the `TwigLexer` that highlights unlexed data with the
    `HtmlLexer`.

    .. versionadded:: 2.0
    """
    name = 'HTML+Twig'
    aliases = ['html+twig']
    filenames = ['*.twig']
    mimetypes = ['text/html+twig']

    def __init__(self, **options):
        super(TwigHtmlLexer, self).__init__(HtmlLexer, TwigLexer, **options)