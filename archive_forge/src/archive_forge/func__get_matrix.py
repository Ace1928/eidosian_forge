import warnings
import numpy as np
from scipy.sparse import csc_matrix
from ._fortran_format_parser import FortranFormatParser, IntFormat, ExpFormat
def _get_matrix(fid):
    hb = HBFile(fid)
    return hb.read_matrix()