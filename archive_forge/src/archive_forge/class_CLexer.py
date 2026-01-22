import re
from pygments.lexer import RegexLexer, include, bygroups, using, \
from pygments.util import get_bool_opt
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class CLexer(CFamilyLexer):
    """
    For C source code with preprocessor directives.
    """
    name = 'C'
    aliases = ['c']
    filenames = ['*.c', '*.h', '*.idc']
    mimetypes = ['text/x-chdr', 'text/x-csrc']
    priority = 0.1

    def analyse_text(text):
        if re.search('^\\s*#include [<"]', text, re.MULTILINE):
            return 0.1
        if re.search('^\\s*#ifn?def ', text, re.MULTILINE):
            return 0.1