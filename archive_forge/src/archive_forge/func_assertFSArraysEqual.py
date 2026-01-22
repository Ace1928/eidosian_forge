import itertools
import sys
import logging
from .formatstring import fmtstr
from .formatstring import normalize_slice
from .formatstring import FmtStr
from typing import (
def assertFSArraysEqual(a: FSArray, b: FSArray) -> None:
    assert isinstance(a, FSArray)
    assert isinstance(b, FSArray)
    assert a.width == b.width and a.height == b.height, f'fsarray dimensions do not match: {a.shape} {b.shape}'
    for i, (a_row, b_row) in enumerate(zip(a, b)):
        assert a_row == b_row, 'FSArrays differ first on line {}:\n{}'.format(i, FSArray.diff(a, b))