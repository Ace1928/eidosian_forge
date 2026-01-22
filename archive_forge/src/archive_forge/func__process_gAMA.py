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
def _process_gAMA(self, data):
    try:
        self.gamma = struct.unpack('!L', data)[0] / 100000.0
    except struct.error:
        raise FormatError('gAMA chunk has incorrect length.')