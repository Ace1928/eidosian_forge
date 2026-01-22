import io
import sys
import numpy
import struct
import warnings
from enum import IntEnum
def _array_tofile(fid, data):
    fid.write(data.ravel().view('b').data)