import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
def _quote_pattern(self, pattern):
    if pattern.startswith('https://') or pattern.startswith('http://'):
        pattern = '/' + pattern
    last_char = ''
    if pattern[-1] == '?' or pattern[-1] == ';' or pattern[-1] == '$':
        last_char = pattern[-1]
        pattern = pattern[:-1]
    parts = urlparse(pattern)
    pattern = self._unquote(parts.path, ignore='/*$%')
    pattern = quote(pattern, safe='/*%=')
    parts = ParseResult('', '', pattern + last_char, parts.params, parts.query, parts.fragment)
    pattern = urlunparse(parts)
    return pattern