import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def uncurl_quotes(text):
    """
    Replace curly quotation marks with straight equivalents.

        >>> print(uncurl_quotes('\\u201chere\\u2019s a test\\u201d'))
        "here's a test"
    """
    return SINGLE_QUOTE_RE.sub("'", DOUBLE_QUOTE_RE.sub('"', text))