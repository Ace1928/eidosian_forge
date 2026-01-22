import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
def _read_punctuation(self):
    token = None
    resulting_string = self._patterns.punct.read()
    if resulting_string != '':
        if resulting_string == '=':
            token = self._create_token(TOKEN.EQUALS, resulting_string)
        elif resulting_string == '?.':
            token = self._create_token(TOKEN.DOT, resulting_string)
        else:
            token = self._create_token(TOKEN.OPERATOR, resulting_string)
    return token