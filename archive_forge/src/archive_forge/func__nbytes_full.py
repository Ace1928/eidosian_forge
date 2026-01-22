import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
def _nbytes_full(fmt, nlines):
    """Return the number of bytes to read to get every full lines for the
    given parsed fortran format."""
    return (fmt.repeat * fmt.width + 1) * (nlines - 1)