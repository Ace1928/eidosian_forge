import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class LiterateAgdaLexer(LiterateLexer):
    """
    For Literate Agda source.

    Additional options accepted:

    `litstyle`
        If given, must be ``"bird"`` or ``"latex"``.  If not given, the style
        is autodetected: if the first non-whitespace character in the source
        is a backslash or percent character, LaTeX is assumed, else Bird.

    .. versionadded:: 2.0
    """
    name = 'Literate Agda'
    aliases = ['lagda', 'literate-agda']
    filenames = ['*.lagda']
    mimetypes = ['text/x-literate-agda']

    def __init__(self, **options):
        agdalexer = AgdaLexer(**options)
        LiterateLexer.__init__(self, agdalexer, litstyle='latex', **options)