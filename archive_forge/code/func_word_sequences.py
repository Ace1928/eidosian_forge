import re
from pygments.token import  Comment, Operator, Keyword, Name, String, \
from pygments.lexer import RegexLexer, words, bygroups
def word_sequences(tokens):
    return '(' + '|'.join((token.replace(' ', '\\s+') for token in tokens)) + ')\\b'