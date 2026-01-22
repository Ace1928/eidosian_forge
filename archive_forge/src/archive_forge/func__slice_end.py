from __future__ import absolute_import, unicode_literals
import itertools
import warnings
from abc import ABCMeta, abstractmethod
import six
from pybtex import textutils
from pybtex.utils import collect_iterable, deprecated
from pybtex import py3compat
def _slice_end(self, slice_length):
    """
        Return a text consistng of the last slice_length characters
        of this text (with formatting preserved).
        """
    parts = []
    length = 0
    for part in reversed(self.parts):
        if length + len(part) > slice_length:
            parts.append(part[len(part) - (slice_length - length):])
            break
        else:
            parts.append(part)
            length += len(part)
    return self._create_similar(reversed(parts))