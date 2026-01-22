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
def _process_pHYs(self, data):
    self.phys = data
    fmt = '!LLB'
    if len(data) != struct.calcsize(fmt):
        raise FormatError('pHYs chunk has incorrect length.')
    self.x_pixels_per_unit, self.y_pixels_per_unit, unit = struct.unpack(fmt, data)
    self.unit_is_meter = bool(unit)