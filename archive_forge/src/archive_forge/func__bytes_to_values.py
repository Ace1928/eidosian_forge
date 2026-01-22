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
def _bytes_to_values(self, bs, width=None):
    """Convert a packed row of bytes into a row of values.
        Result will be a freshly allocated object,
        not shared with the argument.
        """
    if self.bitdepth == 8:
        return bytearray(bs)
    if self.bitdepth == 16:
        return array('H', struct.unpack(f'!{len(bs) // 2}H', bs))
    assert self.bitdepth < 8
    if width is None:
        width = self.width
    spb = 8 // self.bitdepth
    out = bytearray()
    mask = 2 ** self.bitdepth - 1
    shifts = [self.bitdepth * i for i in reversed(list(range(spb)))]
    for o in bs:
        out.extend([mask & o >> i for i in shifts])
    return out[:width]