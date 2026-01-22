import sys
import warnings
import numpy as np
import scipy.sparse
from ._miobase import (MatFileReader, docfiller, matdims, read_dtype,
from ._mio_utils import squeeze_element, chars_to_strings
from functools import reduce
 Write variables in `mdict` to stream

        Parameters
        ----------
        mdict : mapping
           mapping with method ``items`` return name, contents pairs
           where ``name`` which will appeak in the matlab workspace in
           file load, and ``contents`` is something writeable to a
           matlab file, such as a NumPy array.
        write_header : {None, True, False}
           If True, then write the matlab file header before writing the
           variables. If None (the default) then write the file header
           if we are at position 0 in the stream. By setting False
           here, and setting the stream position to the end of the file,
           you can append variables to a matlab file
        