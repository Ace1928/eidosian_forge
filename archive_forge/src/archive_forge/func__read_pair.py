import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
def _read_pair(self, c, d):
    token = None
    if c == '#' and d == '{':
        token = self._create_token(TOKEN.START_BLOCK, c + d)
    if token is not None:
        self._input.next()
        self._input.next()
    return token