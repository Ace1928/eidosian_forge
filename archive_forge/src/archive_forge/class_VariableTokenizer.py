import re
from pygments.lexer import Lexer
from pygments.token import Token
from pygments.util import text_type
class VariableTokenizer(object):

    def tokenize(self, string, token):
        var = VariableSplitter(string, identifiers='$@%&')
        if var.start < 0 or token in (COMMENT, ERROR):
            yield (string, token)
            return
        for value, token in self._tokenize(var, string, token):
            if value:
                yield (value, token)

    def _tokenize(self, var, string, orig_token):
        before = string[:var.start]
        yield (before, orig_token)
        yield (var.identifier + '{', SYNTAX)
        for value, token in self.tokenize(var.base, VARIABLE):
            yield (value, token)
        yield ('}', SYNTAX)
        if var.index:
            yield ('[', SYNTAX)
            for value, token in self.tokenize(var.index, VARIABLE):
                yield (value, token)
            yield (']', SYNTAX)
        for value, token in self.tokenize(string[var.end:], orig_token):
            yield (value, token)