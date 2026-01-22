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
def asRGBA8(self):
    """
        Return the image data as RGBA pixels with 8-bits per sample.
        This method is similar to :meth:`asRGB8` and :meth:`asRGBA`:
        The result pixels have an alpha channel, *and*
        values are rescaled to the range 0 to 255.
        The alpha channel is synthesized if necessary
        (with a small speed penalty).
        """
    return self._as_rescale(self.asRGBA, 8)