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
def adjust_line_length(cls, line: List['Segment'], length: int, style: Optional[Style]=None, pad: bool=True) -> List['Segment']:
    """Adjust a line to a given width (cropping or padding as required).

        Args:
            segments (Iterable[Segment]): A list of segments in a single line.
            length (int): The desired width of the line.
            style (Style, optional): The style of padding if used (space on the end). Defaults to None.
            pad (bool, optional): Pad lines with spaces if they are shorter than `length`. Defaults to True.

        Returns:
            List[Segment]: A line of segments with the desired length.
        """
    line_length = sum((segment.cell_length for segment in line))
    new_line: List[Segment]
    if line_length < length:
        if pad:
            new_line = line + [cls(' ' * (length - line_length), style)]
        else:
            new_line = line[:]
    elif line_length > length:
        new_line = []
        append = new_line.append
        line_length = 0
        for segment in line:
            segment_length = segment.cell_length
            if line_length + segment_length < length or segment.control:
                append(segment)
                line_length += segment_length
            else:
                text, segment_style, _ = segment
                text = set_cell_size(text, length - line_length)
                append(cls(text, segment_style))
                break
    else:
        new_line = line[:]
    return new_line