import re
from pygments.lexer import RegexLexer, ExtendedRegexLexer, include, bygroups, \
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.util import shebang_matches
def _build_word_match(words, boundary_regex_fragment=None, prefix='', suffix=''):
    if boundary_regex_fragment is None:
        return '\\b(' + prefix + '|'.join((re.escape(x) for x in words)) + suffix + ')\\b'
    else:
        return '(?<!' + boundary_regex_fragment + ')' + prefix + '(' + '|'.join((re.escape(x) for x in words)) + ')' + suffix + '(?!' + boundary_regex_fragment + ')'