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
def _get_global_header(im, info):
    """Return a list of strings representing a GIF header"""
    version = b'87a'
    if im.info.get('version') == b'89a' or (info and ('transparency' in info or info.get('loop') is not None or info.get('duration') or info.get('comment'))):
        version = b'89a'
    background = _get_background(im, info.get('background'))
    palette_bytes = _get_palette_bytes(im)
    color_table_size = _get_color_table_size(palette_bytes)
    header = [b'GIF' + version + o16(im.size[0]) + o16(im.size[1]), o8(color_table_size + 128), o8(background) + o8(0), _get_header_palette(palette_bytes)]
    if info.get('loop') is not None:
        header.append(b'!' + o8(255) + o8(11) + b'NETSCAPE2.0' + o8(3) + o8(1) + o16(info['loop']) + o8(0))
    if info.get('comment'):
        comment_block = b'!' + o8(254)
        comment = info['comment']
        if isinstance(comment, str):
            comment = comment.encode()
        for i in range(0, len(comment), 255):
            subblock = comment[i:i + 255]
            comment_block += o8(len(subblock)) + subblock
        comment_block += o8(0)
        header.append(comment_block)
    return header