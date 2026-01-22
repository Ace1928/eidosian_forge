import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer, LassoLexer
from pygments.lexers.css import CssLexer
from pygments.lexers.php import PhpLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.perl import PerlLexer
from pygments.lexers.jvm import JavaLexer, TeaLangLexer
from pygments.lexers.data import YamlLexer
from pygments.lexers.sql import SqlLexer
from pygments.lexer import Lexer, DelegatingLexer, RegexLexer, bygroups, \
from pygments.token import Error, Punctuation, Whitespace, \
from pygments.util import html_doctype_matches, looks_like_xml
class SqlJinjaLexer(DelegatingLexer):
    """
    Templated SQL lexer.

    .. versionadded:: 2.13
    """
    name = 'SQL+Jinja'
    aliases = ['sql+jinja']
    filenames = ['*.sql', '*.sql.j2', '*.sql.jinja2']

    def __init__(self, **options):
        super().__init__(SqlLexer, DjangoLexer, **options)

    def analyse_text(text):
        rv = 0.0
        if re.search('\\{\\{\\s*ref\\(.*\\)\\s*\\}\\}', text):
            rv += 0.4
        if re.search('\\{\\{\\s*source\\(.*\\)\\s*\\}\\}', text):
            rv += 0.25
        if re.search('\\{%-?\\s*macro \\w+\\(.*\\)\\s*-?%\\}', text):
            rv += 0.15
        return rv