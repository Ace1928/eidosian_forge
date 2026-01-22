import re
from pygments.lexer import RegexLexer, bygroups, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def _multi_escape(entries):
    return '(%s)' % '|'.join((re.escape(entry) for entry in entries))