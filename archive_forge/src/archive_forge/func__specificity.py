from __future__ import annotations
import codecs
import re
from .structures import ImmutableList
def _specificity(self, value):
    return tuple((x != '*' for x in _mime_split_re.split(value)))