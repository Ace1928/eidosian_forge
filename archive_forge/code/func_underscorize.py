import re
from pygments.lexer import RegexLexer, include, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
def underscorize(words):
    newWords = []
    new = ''
    for word in words:
        for ch in word:
            new += ch + '_?'
        newWords.append(new)
        new = ''
    return '|'.join(newWords)