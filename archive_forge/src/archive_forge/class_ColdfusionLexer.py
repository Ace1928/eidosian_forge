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
class ColdfusionLexer(RegexLexer):
    """
    Coldfusion statements
    """
    name = 'cfstatement'
    aliases = ['cfs']
    filenames = []
    mimetypes = []
    flags = re.IGNORECASE
    tokens = {'root': [('//.*?\\n', Comment.Single), ('/\\*(?:.|\\n)*?\\*/', Comment.Multiline), ('\\+\\+|--', Operator), ('[-+*/^&=!]', Operator), ('<=|>=|<|>|==', Operator), ('mod\\b', Operator), ('(eq|lt|gt|lte|gte|not|is|and|or)\\b', Operator), ('\\|\\||&&', Operator), ('\\?', Operator), ('"', String.Double, 'string'), ("'.*?'", String.Single), ('\\d+', Number), ('(if|else|len|var|xml|default|break|switch|component|property|function|do|try|catch|in|continue|for|return|while|required|any|array|binary|boolean|component|date|guid|numeric|query|string|struct|uuid|case)\\b', Keyword), ('(true|false|null)\\b', Keyword.Constant), ('(application|session|client|cookie|super|this|variables|arguments)\\b', Name.Constant), ('([a-z_$][\\w.]*)(\\s*)(\\()', bygroups(Name.Function, Text, Punctuation)), ('[a-z_$][\\w.]*', Name.Variable), ('[()\\[\\]{};:,.\\\\]', Punctuation), ('\\s+', Text)], 'string': [('""', String.Double), ('#.+?#', String.Interp), ('[^"#]+', String.Double), ('#', String.Double), ('"', String.Double, '#pop')]}