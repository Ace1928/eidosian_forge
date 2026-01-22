from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
@staticmethod
def _read_value(parent: TiffFile | TiffWriter, offset: int, code: int, dtype: int, count: int, valueoffset: int, /) -> Any:
    """Read tag value from file."""
    try:
        valueformat = TIFF.DATA_FORMATS[dtype]
    except KeyError as exc:
        raise TiffFileError(f'<tifffile.TiffTag {code} @{offset}> invalid data type {dtype!r}') from exc
    fh = parent.filehandle
    tiff = parent.tiff
    valuesize = count * struct.calcsize(valueformat)
    if valueoffset < 8 or valueoffset + valuesize > fh.size:
        raise TiffFileError(f'<tifffile.TiffTag {code} @{offset}> invalid value offset {valueoffset}')
    fh.seek(valueoffset)
    if code in TIFF.TAG_READERS:
        readfunc = TIFF.TAG_READERS[code]
        value = readfunc(fh, tiff.byteorder, dtype, count, tiff.offsetsize)
    elif dtype in {1, 2, 7}:
        value = fh.read(valuesize)
        if len(value) != valuesize:
            logger().warning(f'<tifffile.TiffTag {code} @{offset}> could not read all values')
    elif code not in TIFF.TAG_TUPLE and count > 1024:
        value = read_numpy(fh, tiff.byteorder, dtype, count, tiff.offsetsize)
    else:
        fmt = '{}{}{}'.format(tiff.byteorder, count * int(valueformat[0]), valueformat[1])
        value = struct.unpack(fmt, fh.read(valuesize))
    return value