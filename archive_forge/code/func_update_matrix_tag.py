import os
import time
import sys
import zlib
from io import BytesIO
import warnings
import numpy as np
import scipy.sparse
from ._byteordercodes import native_code, swapped_code
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio5_utils import VarReader5
from ._mio5_params import (MatlabObject, MatlabFunction, MDTYPES, NP_TO_MTYPES,
from ._streams import ZlibInputStream
def update_matrix_tag(self, start_pos):
    curr_pos = self.file_stream.tell()
    self.file_stream.seek(start_pos)
    byte_count = curr_pos - start_pos - 8
    if byte_count >= 2 ** 32:
        raise MatWriteError('Matrix too large to save with Matlab 5 format')
    self.mat_tag['byte_count'] = byte_count
    self.write_bytes(self.mat_tag)
    self.file_stream.seek(curr_pos)