import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
def _nlines(fmt, size):
    nlines = size // fmt.repeat
    if nlines * fmt.repeat != size:
        nlines += 1
    return nlines