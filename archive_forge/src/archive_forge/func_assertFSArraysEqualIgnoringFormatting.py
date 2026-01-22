import itertools
import sys
import logging
from .formatstring import fmtstr
from .formatstring import normalize_slice
from .formatstring import FmtStr
from typing import (
def assertFSArraysEqualIgnoringFormatting(a: FSArray, b: FSArray) -> None:
    """Also accepts arrays of strings"""
    assert len(a) == len(b), 'fsarray heights do not match: %s %s \n%s \n%s' % (len(a), len(b), simple_format(a), simple_format(b))
    for i, (a_row, b_row) in enumerate(zip(a, b)):
        a_row = a_row.s if isinstance(a_row, FmtStr) else a_row
        b_row = b_row.s if isinstance(b_row, FmtStr) else b_row
        assert a_row == b_row, 'FSArrays differ first on line %s:\n%s' % (i, FSArray.diff(a, b, ignore_formatting=True))