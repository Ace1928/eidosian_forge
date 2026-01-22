import re
from itertools import chain
from ._base import (
from .exonerate_vulgar import _RE_VULGAR
def _stitch_rows(raw_rows):
    """Stitches together the parsed alignment rows and returns them in a list (PRIVATE)."""
    try:
        max_len = max((len(x) for x in raw_rows))
        for row in raw_rows:
            assert len(row) == max_len
    except AssertionError:
        for idx, row in enumerate(raw_rows):
            if len(row) != max_len:
                assert len(row) + 2 == max_len
                raw_rows[idx] = [' ' * len(row[0])] + row + [' ' * len(row[0])]
    cmbn_rows = []
    for idx, row in enumerate(raw_rows[0]):
        cmbn_row = ''.join((aln_row[idx] for aln_row in raw_rows))
        cmbn_rows.append(cmbn_row)
    if len(cmbn_rows) == 5:
        cmbn_rows[0], cmbn_rows[1] = _flip_codons(cmbn_rows[0], cmbn_rows[1])
        cmbn_rows[4], cmbn_rows[3] = _flip_codons(cmbn_rows[4], cmbn_rows[3])
    return cmbn_rows