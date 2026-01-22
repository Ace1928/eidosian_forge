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
class AntlrPerlLexer(DelegatingLexer):
    """
    `ANTLR`_ with Perl Target

    .. versionadded:: 1.1
    """
    name = 'ANTLR With Perl Target'
    aliases = ['antlr-perl']
    filenames = ['*.G', '*.g']

    def __init__(self, **options):
        super(AntlrPerlLexer, self).__init__(PerlLexer, AntlrLexer, **options)

    def analyse_text(text):
        return AntlrLexer.analyse_text(text) and re.search('^\\s*language\\s*=\\s*Perl5\\s*;', text, re.M)