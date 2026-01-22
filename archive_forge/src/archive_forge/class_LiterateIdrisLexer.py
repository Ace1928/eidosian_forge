import re
from pygments.lexer import Lexer, RegexLexer, bygroups, do_insertions, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class LiterateIdrisLexer(LiterateLexer):
    """
    For Literate Idris (Bird-style or LaTeX) source.

    Additional options accepted:

    `litstyle`
        If given, must be ``"bird"`` or ``"latex"``.  If not given, the style
        is autodetected: if the first non-whitespace character in the source
        is a backslash or percent character, LaTeX is assumed, else Bird.

    .. versionadded:: 2.0
    """
    name = 'Literate Idris'
    aliases = ['lidr', 'literate-idris', 'lidris']
    filenames = ['*.lidr']
    mimetypes = ['text/x-literate-idris']

    def __init__(self, **options):
        hslexer = IdrisLexer(**options)
        LiterateLexer.__init__(self, hslexer, **options)