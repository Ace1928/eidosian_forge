import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
def allowRegExOrXML(self, previous_token):
    return previous_token.type == TOKEN.RESERVED and previous_token.text in {'return', 'case', 'throw', 'else', 'do', 'typeof', 'yield'} or (previous_token.type == TOKEN.END_EXPR and previous_token.text == ')' and (previous_token.opened.previous.type == TOKEN.RESERVED) and (previous_token.opened.previous.text in {'if', 'while', 'for'})) or previous_token.type in self.__regexTokens