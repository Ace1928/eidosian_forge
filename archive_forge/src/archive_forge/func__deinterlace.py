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
def _deinterlace(self, raw):
    """
        Read raw pixel data, undo filters, deinterlace, and flatten.
        Return a single array of values.
        """
    vpr = self.width * self.planes
    vpi = vpr * self.height
    if self.bitdepth > 8:
        a = array('H', [0] * vpi)
    else:
        a = bytearray([0] * vpi)
    source_offset = 0
    for lines in adam7_generate(self.width, self.height):
        recon = None
        for x, y, xstep in lines:
            ppr = int(math.ceil((self.width - x) / float(xstep)))
            row_size = int(math.ceil(self.psize * ppr))
            filter_type = raw[source_offset]
            source_offset += 1
            scanline = raw[source_offset:source_offset + row_size]
            source_offset += row_size
            recon = self.undo_filter(filter_type, scanline, recon)
            flat = self._bytes_to_values(recon, width=ppr)
            if xstep == 1:
                assert x == 0
                offset = y * vpr
                a[offset:offset + vpr] = flat
            else:
                offset = y * vpr + x * self.planes
                end_offset = (y + 1) * vpr
                skip = self.planes * xstep
                for i in range(self.planes):
                    a[offset + i:end_offset:skip] = flat[i::self.planes]
    return a