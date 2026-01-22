from enum import IntEnum
from functools import lru_cache
from itertools import filterfalse
from logging import getLogger
from operator import attrgetter
from typing import (
from .cells import (
from .repr import Result, rich_repr
from .style import Style
@classmethod
@lru_cache(1024 * 16)
def _split_cells(cls, segment: 'Segment', cut: int) -> Tuple['Segment', 'Segment']:
    text, style, control = segment
    _Segment = Segment
    cell_length = segment.cell_length
    if cut >= cell_length:
        return (segment, _Segment('', style, control))
    cell_size = get_character_cell_size
    pos = int(cut / cell_length * (len(text) - 1))
    before = text[:pos]
    cell_pos = cell_len(before)
    if cell_pos == cut:
        return (_Segment(before, style, control), _Segment(text[pos:], style, control))
    while pos < len(text):
        char = text[pos]
        pos += 1
        cell_pos += cell_size(char)
        before = text[:pos]
        if cell_pos == cut:
            return (_Segment(before, style, control), _Segment(text[pos:], style, control))
        if cell_pos > cut:
            return (_Segment(before[:pos - 1] + ' ', style, control), _Segment(' ' + text[pos:], style, control))
    raise AssertionError('Will never reach here')