import re
from pygments.lexer import Lexer, RegexLexer, include, words, do_insertions
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class RdLexer(RegexLexer):
    """
    Pygments Lexer for R documentation (Rd) files

    This is a very minimal implementation, highlighting little more
    than the macros. A description of Rd syntax is found in `Writing R
    Extensions <http://cran.r-project.org/doc/manuals/R-exts.html>`_
    and `Parsing Rd files <developer.r-project.org/parseRd.pdf>`_.

    .. versionadded:: 1.6
    """
    name = 'Rd'
    aliases = ['rd']
    filenames = ['*.Rd']
    mimetypes = ['text/x-r-doc']
    tokens = {'root': [('\\\\[\\\\{}%]', String.Escape), ('%.*$', Comment), ('\\\\(?:cr|l?dots|R|tab)\\b', Keyword.Constant), ('\\\\[a-zA-Z]+\\b', Keyword), ('^\\s*#(?:ifn?def|endif).*\\b', Comment.Preproc), ('[{}]', Name.Builtin), ('[^\\\\%\\n{}]+', Text), ('.', Text)]}