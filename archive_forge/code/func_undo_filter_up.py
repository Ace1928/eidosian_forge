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
def undo_filter_up(filter_unit, scanline, previous, result):
    """Undo up filter."""
    for i in range(len(result)):
        x = scanline[i]
        b = previous[i]
        result[i] = x + b & 255