from __future__ import annotations
import codecs
import re
from .structures import ImmutableList
def _normalize_mime(value):
    return _mime_split_re.split(value.lower())