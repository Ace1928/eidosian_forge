from __future__ import unicode_literals
import bisect
import re
import six
import string
import weakref
from six.moves import range, map
from .selection import SelectionType, SelectionState, PasteMode
from .clipboard import ClipboardData
@property
def text_before_cursor(self):
    return self.text[:self.cursor_position]