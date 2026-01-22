import argparse
import codecs
import logging
import re
import sys
from collections import defaultdict, OrderedDict
from hashlib import sha256
from random import randint, random
def _extract_url_and_title(self, text, start):
    """Extracts the url and (optional) title from the tail of a link"""
    idx = self._find_non_whitespace(text, start + 1)
    if idx == len(text):
        return (None, None, None)
    end_idx = idx
    has_anglebrackets = text[idx] == '<'
    if has_anglebrackets:
        end_idx = self._find_balanced(text, end_idx + 1, '<', '>')
    end_idx = self._find_balanced(text, end_idx, '(', ')')
    match = self._inline_link_title.search(text, idx, end_idx)
    if not match:
        return (None, None, None)
    url, title = (text[idx:match.start()], match.group('title'))
    if has_anglebrackets:
        url = self._strip_anglebrackets.sub('\\1', url)
    return (url, title, end_idx)