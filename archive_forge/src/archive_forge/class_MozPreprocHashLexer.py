import re
from pygments.lexers.html import HtmlLexer, XmlLexer
from pygments.lexers.javascript import JavascriptLexer
from pygments.lexers.css import CssLexer
from pygments.lexer import RegexLexer, DelegatingLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import get_bool_opt, ClassNotFound
class MozPreprocHashLexer(RegexLexer):
    """
    Lexer for Mozilla Preprocessor files (with '#' as the marker).

    Other data is left untouched.

    .. versionadded:: 2.0
    """
    name = 'mozhashpreproc'
    aliases = [name]
    filenames = []
    mimetypes = []
    tokens = {'root': [('^#', Comment.Preproc, ('expr', 'exprstart')), ('.+', Other)], 'exprstart': [('(literal)(.*)', bygroups(Comment.Preproc, Text), '#pop:2'), (words(('define', 'undef', 'if', 'ifdef', 'ifndef', 'else', 'elif', 'elifdef', 'elifndef', 'endif', 'expand', 'filter', 'unfilter', 'include', 'includesubst', 'error')), Comment.Preproc, '#pop')], 'expr': [(words(('!', '!=', '==', '&&', '||')), Operator), ('(defined)(\\()', bygroups(Keyword, Punctuation)), ('\\)', Punctuation), ('[0-9]+', Number.Decimal), ('__\\w+?__', Name.Variable), ('@\\w+?@', Name.Class), ('\\w+', Name), ('\\n', Text, '#pop'), ('\\s+', Text), ('\\S', Punctuation)]}