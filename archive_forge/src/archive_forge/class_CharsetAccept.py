from __future__ import annotations
import codecs
import re
from .structures import ImmutableList
class CharsetAccept(Accept):
    """Like :class:`Accept` but with normalization for charsets."""

    def _value_matches(self, value, item):

        def _normalize(name):
            try:
                return codecs.lookup(name).name
            except LookupError:
                return name.lower()
        return item == '*' or _normalize(value) == _normalize(item)