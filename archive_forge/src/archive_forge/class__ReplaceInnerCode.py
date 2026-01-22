import re
import keyword
from pygments.lexer import DelegatingLexer, Lexer, RegexLexer, include, \
from pygments.util import get_bool_opt, shebang_matches
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments import unistring as uni
class _ReplaceInnerCode(DelegatingLexer):

    def __init__(self, **options):
        super().__init__(pylexer, _PythonConsoleLexerBase, Other.Code, **options)