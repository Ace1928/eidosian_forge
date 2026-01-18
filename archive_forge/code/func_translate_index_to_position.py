from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
def translate_index_to_position(self, index):
    """
        Given an index for the text, return the corresponding (row, col) tuple.
        (0-based. Returns (0, 0) for index=0.)
        """
    row, row_index = self._find_line_start_index(index)
    col = index - row_index
    return (row, col)