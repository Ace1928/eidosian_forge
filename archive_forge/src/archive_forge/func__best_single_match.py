from __future__ import annotations
import codecs
import re
from .structures import ImmutableList
def _best_single_match(self, match):
    for client_item, quality in self:
        if self._value_matches(match, client_item):
            return (client_item, quality)
    return None