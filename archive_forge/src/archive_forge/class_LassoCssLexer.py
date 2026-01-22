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
class LassoCssLexer(DelegatingLexer):
    """
    Subclass of the `LassoLexer` which highlights unhandled data with the
    `CssLexer`.

    .. versionadded:: 1.6
    """
    name = 'CSS+Lasso'
    aliases = ['css+lasso']
    alias_filenames = ['*.css']
    mimetypes = ['text/css+lasso']

    def __init__(self, **options):
        options['requiredelimiters'] = True
        super(LassoCssLexer, self).__init__(CssLexer, LassoLexer, **options)

    def analyse_text(text):
        rv = LassoLexer.analyse_text(text) - 0.05
        if re.search('\\w+:.+?;', text):
            rv += 0.1
        if 'padding:' in text:
            rv += 0.1
        return rv