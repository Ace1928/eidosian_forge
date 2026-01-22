from the public API.  This format is called packed.  When packed,
import io
import itertools
import math
import operator
import struct
import sys
import zlib
import warnings
from array import array
from functools import reduce
from pygame.tests.test_utils import tostring
fromarray = from_array
import tempfile
import unittest
def asFloat(self, maxval=1.0):
    """Return image pixels as per :meth:`asDirect` method, but scale
        all pixel values to be floating point values between 0.0 and
        *maxval*.
        """
    x, y, pixels, info = self.asDirect()
    sourcemaxval = 2 ** info['bitdepth'] - 1
    del info['bitdepth']
    info['maxval'] = float(maxval)
    factor = float(maxval) / float(sourcemaxval)

    def iterfloat():
        for row in pixels:
            yield map(factor.__mul__, row)
    return (x, y, iterfloat(), info)