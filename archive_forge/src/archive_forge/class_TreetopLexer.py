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
class TreetopLexer(DelegatingLexer):
    """
    A lexer for `Treetop <http://treetop.rubyforge.org/>`_ grammars.

    .. versionadded:: 1.6
    """
    name = 'Treetop'
    aliases = ['treetop']
    filenames = ['*.treetop', '*.tt']

    def __init__(self, **options):
        super(TreetopLexer, self).__init__(RubyLexer, TreetopBaseLexer, **options)