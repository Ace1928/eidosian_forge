import pickle
import hashlib
import sys
import types
import struct
import io
import decimal
class _ConsistentSet(object):
    """ Class used to ensure the hash of Sets is preserved
        whatever the order of its items.
    """

    def __init__(self, set_sequence):
        try:
            self._sequence = sorted(set_sequence)
        except (TypeError, decimal.InvalidOperation):
            self._sequence = sorted((hash(e) for e in set_sequence))