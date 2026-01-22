import re
from ..core.inputscanner import InputScanner
from ..core.tokenizer import TokenTypes as BaseTokenTypes
from ..core.tokenizer import Tokenizer as BaseTokenizer
from ..core.tokenizer import TokenizerPatterns as BaseTokenizerPatterns
from ..core.directives import Directives
from ..core.pattern import Pattern
from ..core.templatablepattern import TemplatablePattern
def _read_regexp(self, c, previous_token):
    if c == '/' and self.allowRegExOrXML(previous_token):
        resulting_string = self._input.next()
        esc = False
        in_char_class = False
        while self._input.hasNext() and (esc or in_char_class or self._input.peek() != c) and (not self._input.testChar(self.acorn.newline)):
            resulting_string += self._input.peek()
            if not esc:
                esc = self._input.peek() == '\\'
                if self._input.peek() == '[':
                    in_char_class = True
                elif self._input.peek() == ']':
                    in_char_class = False
            else:
                esc = False
            self._input.next()
        if self._input.peek() == c:
            resulting_string += self._input.next()
            if c == '/':
                resulting_string += self._input.read(self.acorn.identifier)
        return self._create_token(TOKEN.STRING, resulting_string)
    return None