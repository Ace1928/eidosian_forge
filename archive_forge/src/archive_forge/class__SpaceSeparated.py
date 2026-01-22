import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
class _SpaceSeparated(object):
    """Namespace for conversion methods for space-separated lists as tuples."""
    _has_space = re.compile('\\s')

    @staticmethod
    def from_str(s):
        """Returns the values in s as a tuple (empty if only whitespace)."""
        return tuple((v for v in (s or '').split() if v))

    @classmethod
    def to_str(cls, seq):
        """Returns the sequence as a space-separated string (None if empty)."""
        l = list(seq)
        if not l:
            return None
        tmp = []
        for s in l:
            if cls._has_space.search(s):
                raise MachineReadableFormatError('values must not contain whitespace')
            s = s.strip()
            if not s:
                raise MachineReadableFormatError('values must not be empty')
            tmp.append(s)
        return ' '.join(tmp)