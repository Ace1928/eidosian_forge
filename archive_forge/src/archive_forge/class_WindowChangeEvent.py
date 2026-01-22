import codecs
import itertools
import sys
from enum import Enum, auto
from typing import Optional, List, Sequence, Union
from .termhelpers import Termmode
from .curtsieskeys import CURTSIES_NAMES as special_curtsies_names
class WindowChangeEvent(Event):

    def __init__(self, rows: int, columns: int, cursor_dy: Optional[int]=None) -> None:
        self.rows = rows
        self.columns = columns
        self.cursor_dy = cursor_dy
    x = width = property(lambda self: self.columns)
    y = height = property(lambda self: self.rows)

    def __repr__(self) -> str:
        return '<WindowChangeEvent (%d, %d)%s>' % (self.rows, self.columns, '' if self.cursor_dy is None else ' cursor_dy: %d' % self.cursor_dy)

    @property
    def name(self) -> str:
        return '<WindowChangeEvent>'