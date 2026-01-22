from __future__ import annotations
import codecs
import re
from .structures import ImmutableList
def _normalize_lang(value):
    """Process a language tag for matching."""
    return _locale_delim_re.split(value.lower())