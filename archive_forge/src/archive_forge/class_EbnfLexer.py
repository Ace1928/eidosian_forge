import re
from pygments.lexer import RegexLexer, DelegatingLexer, \
from pygments.token import Punctuation, Other, Text, Comment, Operator, \
from pygments.lexers.jvm import JavaLexer
from pygments.lexers.c_cpp import CLexer, CppLexer
from pygments.lexers.objective import ObjectiveCLexer
from pygments.lexers.d import DLexer
from pygments.lexers.dotnet import CSharpLexer
from pygments.lexers.ruby import RubyLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.perl import PerlLexer
class EbnfLexer(RegexLexer):
    """
    Lexer for `ISO/IEC 14977 EBNF
    <http://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_Form>`_
    grammars.

    .. versionadded:: 2.0
    """
    name = 'EBNF'
    aliases = ['ebnf']
    filenames = ['*.ebnf']
    mimetypes = ['text/x-ebnf']
    tokens = {'root': [include('whitespace'), include('comment_start'), include('identifier'), ('=', Operator, 'production')], 'production': [include('whitespace'), include('comment_start'), include('identifier'), ('"[^"]*"', String.Double), ("'[^']*'", String.Single), ('(\\?[^?]*\\?)', Name.Entity), ('[\\[\\]{}(),|]', Punctuation), ('-', Operator), (';', Punctuation, '#pop'), ('\\.', Punctuation, '#pop')], 'whitespace': [('\\s+', Text)], 'comment_start': [('\\(\\*', Comment.Multiline, 'comment')], 'comment': [('[^*)]', Comment.Multiline), include('comment_start'), ('\\*\\)', Comment.Multiline, '#pop'), ('[*)]', Comment.Multiline)], 'identifier': [('([a-zA-Z][\\w \\-]*)', Keyword)]}