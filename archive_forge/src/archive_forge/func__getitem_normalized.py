import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def _getitem_normalized(self, index: Union[int, slice]) -> 'FmtStr':
    """Builds the more compact fmtstrs by using fromstr( of the control sequences)"""
    index = normalize_slice(len(self), index)
    counter = 0
    output = ''
    for fs in self.chunks:
        if index.start < counter + len(fs) and index.stop > counter:
            s_part = fs.s[max(0, index.start - counter):index.stop - counter]
            piece = Chunk(s_part, fs.atts).color_str
            output += piece
        counter += len(fs)
        if index.stop < counter:
            break
    return fmtstr(output)