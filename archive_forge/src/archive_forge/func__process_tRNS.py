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
def _process_tRNS(self, data):
    self.trns = data
    if self.colormap:
        if not self.plte:
            warnings.warn('PLTE chunk is required before tRNS chunk.')
        elif len(data) > len(self.plte) / 3:
            raise FormatError('tRNS chunk is too long.')
    else:
        if self.alpha:
            raise FormatError(f'tRNS chunk is not valid with colour type {self.color_type}.')
        try:
            self.transparent = struct.unpack(f'!{self.color_planes}H', data)
        except struct.error:
            raise FormatError('tRNS chunk has incorrect length.')