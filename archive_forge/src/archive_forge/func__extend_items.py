import sys
import types
from array import array
from collections import abc
from ._abc import MultiMapping, MutableMultiMapping
def _extend_items(self, items):
    for identity, key, value in items:
        self.add(key, value)