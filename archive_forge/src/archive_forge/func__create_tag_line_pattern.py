import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Comment, Name, Text, Punctuation, String, Keyword
def _create_tag_line_pattern(inner_pattern, ignore_case=False):
    return ('(?i)' if ignore_case else '') + '^(##)( *)' + inner_pattern + '( *)(:)( *)(.*?)( *)$'