import sys
import types
from array import array
from collections import abc
from ._abc import MultiMapping, MutableMultiMapping
class CIMultiDict(MultiDict):
    """Dictionary with the support for duplicate case-insensitive keys."""

    def _title(self, key):
        return key.title()