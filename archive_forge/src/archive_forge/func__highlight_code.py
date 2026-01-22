import re
from pygments.lexers import guess_lexer, get_lexer_by_name
from pygments.lexer import RegexLexer, bygroups, default, include
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import ClassNotFound
def _highlight_code(self, match):
    code = match.group(1)
    try:
        if self.body_lexer:
            lexer = get_lexer_by_name(self.body_lexer)
        else:
            lexer = guess_lexer(code.strip())
    except ClassNotFound:
        lexer = get_lexer_by_name('text')
    yield from lexer.get_tokens_unprocessed(code)