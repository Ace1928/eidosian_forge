import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class HspecLexer(HaskellLexer):
    """
    A Haskell lexer with support for Hspec constructs.

    .. versionadded:: 2.4.0
    """
    name = 'Hspec'
    aliases = ['hspec']
    filenames = ['*Spec.hs']
    mimetypes = []
    tokens = {'root': [('(it)(\\s*)("[^"]*")', bygroups(Text, Whitespace, String.Doc)), ('(describe)(\\s*)("[^"]*")', bygroups(Text, Whitespace, String.Doc)), ('(context)(\\s*)("[^"]*")', bygroups(Text, Whitespace, String.Doc)), inherit]}