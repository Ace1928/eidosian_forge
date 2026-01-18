import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def width_aware_slice(self, index: Union[int, slice]) -> 'FmtStr':
    """Slice based on the number of columns it would take to display the substring."""
    if wcswidth(self.s, None) == -1:
        raise ValueError('bad values for width aware slicing')
    index = normalize_slice(self.width, index)
    counter = 0
    parts = []
    for chunk in self.chunks:
        if index.start < counter + chunk.width and index.stop > counter:
            start = max(0, index.start - counter)
            end = min(index.stop - counter, chunk.width)
            if end - start == chunk.width:
                parts.append(chunk)
            else:
                s_part = width_aware_slice(chunk.s, max(0, index.start - counter), index.stop - counter)
                parts.append(Chunk(s_part, chunk.atts))
        counter += chunk.width
        if index.stop < counter:
            break
    return FmtStr(*parts) if parts else fmtstr('')