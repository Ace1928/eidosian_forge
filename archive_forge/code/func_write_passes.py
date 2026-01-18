import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def write_passes(self, outfile, rows):
    """
        Write a PNG image to the output file.

        Most users are expected to find the :meth:`write` or
        :meth:`write_array` method more convenient.

        The rows should be given to this method in the order that
        they appear in the output file.
        For straightlaced images, this is the usual top to bottom ordering.
        For interlaced images the rows should have been interlaced before
        passing them to this function.

        `rows` should be an iterable that yields each row
        (each row being a sequence of values).
        """
    if self.rescale:
        rows = rescale_rows(rows, self.rescale)
    if self.bitdepth < 8:
        rows = pack_rows(rows, self.bitdepth)
    elif self.bitdepth == 16:
        rows = unpack_rows(rows)
    return self.write_packed(outfile, rows)