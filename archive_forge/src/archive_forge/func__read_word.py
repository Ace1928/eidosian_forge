import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
def _read_word(self, previous_token):
    resulting_string = self._patterns.identifier.read()
    if bool(resulting_string):
        resulting_string = re.sub(self.acorn.allLineBreaks, '\n', resulting_string)
        if not (previous_token.type == TOKEN.DOT or (previous_token.type == TOKEN.RESERVED and (previous_token.text == 'set' or previous_token.text == 'get'))) and reserved_word_pattern.match(resulting_string):
            if (resulting_string == 'in' or resulting_string == 'of') and (previous_token.type == TOKEN.WORD or previous_token.type == TOKEN.STRING):
                return self._create_token(TOKEN.OPERATOR, resulting_string)
            return self._create_token(TOKEN.RESERVED, resulting_string)
        return self._create_token(TOKEN.WORD, resulting_string)
    resulting_string = self._patterns.number.read()
    if resulting_string != '':
        return self._create_token(TOKEN.WORD, resulting_string)