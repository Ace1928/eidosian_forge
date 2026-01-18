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
def write_preamble(self, outfile):
    outfile.write(signature)
    write_chunk(outfile, b'IHDR', struct.pack('!2I5B', self.width, self.height, self.bitdepth, self.color_type, 0, 0, self.interlace))
    if self.gamma is not None:
        write_chunk(outfile, b'gAMA', struct.pack('!L', int(round(self.gamma * 100000.0))))
    if self.rescale:
        write_chunk(outfile, b'sBIT', struct.pack(f'{(self.planes, *[s[0] for s in self.rescale])}B'))
    if self.palette:
        p, t = make_palette_chunks(self.palette)
        write_chunk(outfile, b'PLTE', p)
        if t:
            write_chunk(outfile, b'tRNS', t)
    if self.transparent is not None:
        if self.greyscale:
            fmt = '!1H'
        else:
            fmt = '!3H'
        write_chunk(outfile, b'tRNS', struct.pack(fmt, *self.transparent))
    if self.background is not None:
        if self.greyscale:
            fmt = '!1H'
        else:
            fmt = '!3H'
        write_chunk(outfile, b'bKGD', struct.pack(fmt, *self.background))
    if self.x_pixels_per_unit is not None and self.y_pixels_per_unit is not None:
        tup = (self.x_pixels_per_unit, self.y_pixels_per_unit, int(self.unit_is_meter))
        write_chunk(outfile, b'pHYs', struct.pack('!LLB', *tup))