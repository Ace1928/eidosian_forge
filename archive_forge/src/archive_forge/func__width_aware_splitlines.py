import re
from cwcwidth import wcswidth, wcwidth
from itertools import chain
from typing import (
from .escseqparse import parse, remove_ansi
from .termformatconstants import (
def _width_aware_splitlines(self, columns: int) -> Iterator['FmtStr']:
    splitter = self.chunks[0].splitter()
    chunks_of_line = []
    width_of_line = 0
    for source_chunk in self.chunks:
        splitter.reinit(source_chunk)
        while True:
            request = splitter.request(columns - width_of_line)
            if request is None:
                break
            w, new_chunk = request
            chunks_of_line.append(new_chunk)
            width_of_line += w
            if width_of_line == columns:
                yield FmtStr(*chunks_of_line)
                del chunks_of_line[:]
                width_of_line = 0
    if chunks_of_line:
        yield FmtStr(*chunks_of_line)