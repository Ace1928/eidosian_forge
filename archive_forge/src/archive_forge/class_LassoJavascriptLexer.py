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
class LassoJavascriptLexer(DelegatingLexer):
    """
    Subclass of the `LassoLexer` which highlights unhandled data with the
    `JavascriptLexer`.

    .. versionadded:: 1.6
    """
    name = 'JavaScript+Lasso'
    aliases = ['js+lasso', 'javascript+lasso']
    alias_filenames = ['*.js']
    mimetypes = ['application/x-javascript+lasso', 'text/x-javascript+lasso', 'text/javascript+lasso']

    def __init__(self, **options):
        options['requiredelimiters'] = True
        super(LassoJavascriptLexer, self).__init__(JavascriptLexer, LassoLexer, **options)

    def analyse_text(text):
        rv = LassoLexer.analyse_text(text) - 0.05
        return rv