from __future__ import annotations
import itertools
import math
import os
import subprocess
from enum import IntEnum
from . import (
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16
def _write_local_header(fp, im, offset, flags):
    try:
        transparency = im.encoderinfo['transparency']
    except KeyError:
        transparency = None
    if 'duration' in im.encoderinfo:
        duration = int(im.encoderinfo['duration'] / 10)
    else:
        duration = 0
    disposal = int(im.encoderinfo.get('disposal', 0))
    if transparency is not None or duration != 0 or disposal:
        packed_flag = 1 if transparency is not None else 0
        packed_flag |= disposal << 2
        fp.write(b'!' + o8(249) + o8(4) + o8(packed_flag) + o16(duration) + o8(transparency or 0) + o8(0))
    include_color_table = im.encoderinfo.get('include_color_table')
    if include_color_table:
        palette_bytes = _get_palette_bytes(im)
        color_table_size = _get_color_table_size(palette_bytes)
        if color_table_size:
            flags = flags | 128
            flags = flags | color_table_size
    fp.write(b',' + o16(offset[0]) + o16(offset[1]) + o16(im.size[0]) + o16(im.size[1]) + o8(flags))
    if include_color_table and color_table_size:
        fp.write(_get_header_palette(palette_bytes))
    fp.write(o8(8))