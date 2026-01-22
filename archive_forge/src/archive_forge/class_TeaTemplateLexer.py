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
class TeaTemplateLexer(DelegatingLexer):
    """
    Lexer for `Tea Templates <http://teatrove.org/>`_.

    .. versionadded:: 1.5
    """
    name = 'Tea'
    aliases = ['tea']
    filenames = ['*.tea']
    mimetypes = ['text/x-tea']

    def __init__(self, **options):
        super(TeaTemplateLexer, self).__init__(XmlLexer, TeaTemplateRootLexer, **options)

    def analyse_text(text):
        rv = TeaLangLexer.analyse_text(text) - 0.01
        if looks_like_xml(text):
            rv += 0.4
        if '<%' in text and '%>' in text:
            rv += 0.1
        return rv