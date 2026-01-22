import re
from ..core.inputscanner import InputScanner
from ..core.token import Token
from ..core.tokenstream import TokenStream
from ..core.pattern import Pattern
from ..core.whitespacepattern import WhitespacePattern
def _is_first_token(self):
    return self.__tokens.isEmpty()