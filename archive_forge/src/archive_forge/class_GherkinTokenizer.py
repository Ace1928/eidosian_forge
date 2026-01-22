import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class GherkinTokenizer(object):
    _gherkin_prefix = re.compile('^(Given|When|Then|And) ', re.IGNORECASE)

    def tokenize(self, value, token):
        match = self._gherkin_prefix.match(value)
        if not match:
            return [(value, token)]
        end = match.end()
        return [(value[:end], GHERKIN), (value[end:], token)]