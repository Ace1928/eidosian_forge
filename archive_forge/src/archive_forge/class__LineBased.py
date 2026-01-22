import collections
import itertools
import logging
import io
import re
from debian._deb822_repro import (
from debian.deb822 import RestrictedField, RestrictedFieldError
class _LineBased(object):
    """Namespace for conversion methods for line-based lists as tuples."""

    @staticmethod
    def from_str(s):
        """Returns the lines in 's', with whitespace stripped, as a tuple."""
        return tuple((v for v in (line.strip() for line in (s or '').strip().splitlines()) if v))

    @staticmethod
    def to_str(seq):
        """Returns the sequence as a string with each element on its own line.

        If 'seq' has one element, the result will be on a single line.
        Otherwise, the first line will be blank.
        """
        l = list(seq)
        if not l:
            return None

        def process_and_validate(s):
            s = s.strip()
            if not s:
                raise MachineReadableFormatError('values must not be empty')
            if '\n' in s:
                raise MachineReadableFormatError('values must not contain newlines')
            return s
        if len(l) == 1:
            return process_and_validate(l[0])
        tmp = ['']
        for s in l:
            tmp.append(' ' + process_and_validate(s))
        return '\n'.join(tmp)