import codecs
import html
import re
import warnings
import ftfy
from ftfy.chardata import (
from ftfy.badness import is_bad
def _unescape_fixup(match):
    """
    Replace one matched HTML entity with the character it represents,
    if possible.
    """
    text = match.group(0)
    if text in HTML_ENTITIES:
        return HTML_ENTITIES[text]
    elif text.startswith('&#'):
        unescaped = html.unescape(text)
        if ';' in unescaped:
            return text
        else:
            return unescaped
    else:
        return text