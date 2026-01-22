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
class DjangoLexer(RegexLexer):
    """
    Generic `django <http://www.djangoproject.com/documentation/templates/>`_
    and `jinja <http://wsgiarea.pocoo.org/jinja/>`_ template lexer.

    It just highlights django/jinja code between the preprocessor directives,
    other data is left untouched by the lexer.
    """
    name = 'Django/Jinja'
    aliases = ['django', 'jinja']
    mimetypes = ['application/x-django-templating', 'application/x-jinja']
    flags = re.M | re.S
    tokens = {'root': [('[^{]+', Other), ('\\{\\{', Comment.Preproc, 'var'), ('\\{[*#].*?[*#]\\}', Comment), ('(\\{%)(-?\\s*)(comment)(\\s*-?)(%\\})(.*?)(\\{%)(-?\\s*)(endcomment)(\\s*-?)(%\\})', bygroups(Comment.Preproc, Text, Keyword, Text, Comment.Preproc, Comment, Comment.Preproc, Text, Keyword, Text, Comment.Preproc)), ('(\\{%)(-?\\s*)(raw)(\\s*-?)(%\\})(.*?)(\\{%)(-?\\s*)(endraw)(\\s*-?)(%\\})', bygroups(Comment.Preproc, Text, Keyword, Text, Comment.Preproc, Text, Comment.Preproc, Text, Keyword, Text, Comment.Preproc)), ('(\\{%)(-?\\s*)(filter)(\\s+)([a-zA-Z_]\\w*)', bygroups(Comment.Preproc, Text, Keyword, Text, Name.Function), 'block'), ('(\\{%)(-?\\s*)([a-zA-Z_]\\w*)', bygroups(Comment.Preproc, Text, Keyword), 'block'), ('\\{', Other)], 'varnames': [('(\\|)(\\s*)([a-zA-Z_]\\w*)', bygroups(Operator, Text, Name.Function)), ('(is)(\\s+)(not)?(\\s+)?([a-zA-Z_]\\w*)', bygroups(Keyword, Text, Keyword, Text, Name.Function)), ('(_|true|false|none|True|False|None)\\b', Keyword.Pseudo), ('(in|as|reversed|recursive|not|and|or|is|if|else|import|with(?:(?:out)?\\s*context)?|scoped|ignore\\s+missing)\\b', Keyword), ('(loop|block|super|forloop)\\b', Name.Builtin), ('[a-zA-Z_][\\w-]*', Name.Variable), ('\\.\\w+', Name.Variable), (':?"(\\\\\\\\|\\\\"|[^"])*"', String.Double), (":?'(\\\\\\\\|\\\\'|[^'])*'", String.Single), ('([{}()\\[\\]+\\-*/,:~]|[><=]=?)', Operator), ('[0-9](\\.[0-9]*)?(eE[+-][0-9])?[flFLdD]?|0[xX][0-9a-fA-F]+[Ll]?', Number)], 'var': [('\\s+', Text), ('(-?)(\\}\\})', bygroups(Text, Comment.Preproc), '#pop'), include('varnames')], 'block': [('\\s+', Text), ('(-?)(%\\})', bygroups(Text, Comment.Preproc), '#pop'), include('varnames'), ('.', Punctuation)]}

    def analyse_text(text):
        rv = 0.0
        if re.search('\\{%\\s*(block|extends)', text) is not None:
            rv += 0.4
        if re.search('\\{%\\s*if\\s*.*?%\\}', text) is not None:
            rv += 0.1
        if re.search('\\{\\{.*?\\}\\}', text) is not None:
            rv += 0.1
        return rv