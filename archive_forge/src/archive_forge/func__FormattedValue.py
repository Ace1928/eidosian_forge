import ast
import io
import sys
import tokenize
def _FormattedValue(self, t):
    self.write('f')
    string = io.StringIO()
    self._fstring_FormattedValue(t, string.write)
    self.write(repr(string.getvalue()))