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
def _process_bKGD(self, data):
    try:
        if self.colormap:
            if not self.plte:
                warnings.warn('PLTE chunk is required before bKGD chunk.')
            self.background = struct.unpack('B', data)
        else:
            self.background = struct.unpack(f'!{self.color_planes}H', data)
    except struct.error:
        raise FormatError('bKGD chunk has incorrect length.')