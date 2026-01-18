import html
import json
import re
import warnings
from html.parser import HTMLParser
from urllib.parse import parse_qsl, quote, unquote, urlencode, urlsplit, urlunsplit
from django.utils.deprecation import RemovedInDjango60Warning
from django.utils.encoding import punycode
from django.utils.functional import Promise, keep_lazy, keep_lazy_text
from django.utils.http import RFC3986_GENDELIMS, RFC3986_SUBDELIMS
from django.utils.regex_helper import _lazy_re_compile
from django.utils.safestring import SafeData, SafeString, mark_safe
from django.utils.text import normalize_newlines
def trim_punctuation(self, word):
    """
        Trim trailing and wrapping punctuation from `word`. Return the items of
        the new state.
        """
    lead, middle, trail = ('', word, '')
    trimmed_something = True
    while trimmed_something:
        trimmed_something = False
        for opening, closing in self.wrapping_punctuation:
            if middle.startswith(opening):
                middle = middle.removeprefix(opening)
                lead += opening
                trimmed_something = True
            if middle.endswith(closing) and middle.count(closing) == middle.count(opening) + 1:
                middle = middle.removesuffix(closing)
                trail = closing + trail
                trimmed_something = True
        middle_unescaped = html.unescape(middle)
        stripped = middle_unescaped.rstrip(self.trailing_punctuation_chars)
        if middle_unescaped != stripped:
            punctuation_count = len(middle_unescaped) - len(stripped)
            trail = middle[-punctuation_count:] + trail
            middle = middle[:-punctuation_count]
            trimmed_something = True
    return (lead, middle, trail)