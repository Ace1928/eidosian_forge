import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
def _prepare_pattern_for_regex(self, pattern):
    """Return equivalent regex pattern for the given URL pattern."""
    pattern = re.sub('\\*+', '*', pattern)
    s = re.split('(\\*|\\$$)', pattern)
    for index, substr in enumerate(s):
        if substr not in _WILDCARDS:
            s[index] = re.escape(substr)
        elif s[index] == '*':
            s[index] = '.*?'
    pattern = ''.join(s)
    return pattern