import re
from ..core.pattern import Pattern
def __set_whitespace_patterns(self, whitespace_chars, newline_chars):
    whitespace_chars += '\\t '
    newline_chars += '\\n\\r'
    self._match_pattern = self._input.get_regexp('[' + whitespace_chars + newline_chars + ']+')
    self._newline_regexp = self._input.get_regexp('\\r\\n|[' + newline_chars + ']')