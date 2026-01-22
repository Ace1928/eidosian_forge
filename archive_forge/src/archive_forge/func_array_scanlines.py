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
def array_scanlines(self, pixels):
    """
        Generates rows (each a sequence of values) from
        a single array of values.
        """
    vpr = self.width * self.planes
    stop = 0
    for y in range(self.height):
        start = stop
        stop = start + vpr
        yield pixels[start:stop]