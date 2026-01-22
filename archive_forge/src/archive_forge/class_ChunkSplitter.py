import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
class ChunkSplitter:
    """
    View of a Chunk for breaking it into smaller Chunks.
    """

    def __init__(self, chunk: Chunk) -> None:
        self.reinit(chunk)

    def reinit(self, chunk: Chunk) -> None:
        """Reuse an existing Splitter instance for speed."""
        self.chunk = chunk
        self.internal_offset = 0
        self.internal_width = 0
        divides = [0]
        for c in self.chunk.s:
            divides.append(divides[-1] + wcwidth(c))
        self.divides = divides

    def request(self, max_width: int) -> Optional[Tuple[int, Chunk]]:
        """Requests a sub-chunk of max_width or shorter. Returns None if no chunks left."""
        if max_width < 1:
            raise ValueError('requires positive integer max_width')
        s = self.chunk.s
        length = len(s)
        if self.internal_offset == len(s):
            return None
        width = 0
        start_offset = i = self.internal_offset
        replacement_char = ' '
        while True:
            w = wcswidth(s[i], None)
            if width + w > max_width:
                self.internal_offset = i
                self.internal_width += width
                if width < max_width:
                    assert width + 1 == max_width, 'unicode character width of more than 2!?!'
                    assert w == 2, 'unicode character of width other than 2?'
                    return (width + 1, Chunk(s[start_offset:self.internal_offset] + replacement_char, atts=self.chunk.atts))
                return (width, Chunk(s[start_offset:self.internal_offset], atts=self.chunk.atts))
            width += w
            if i + 1 == length:
                self.internal_offset = i + 1
                self.internal_width += width
                return (width, Chunk(s[start_offset:self.internal_offset], atts=self.chunk.atts))
            i += 1