from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
def _GetTermSizeWindows():
    """Returns the Windows terminal x and y dimensions."""
    import struct
    from ctypes import create_string_buffer
    from ctypes import windll
    h = windll.kernel32.GetStdHandle(-12)
    csbi = create_string_buffer(22)
    if not windll.kernel32.GetConsoleScreenBufferInfo(h, csbi):
        return None
    unused_bufx, unused_bufy, unused_curx, unused_cury, unused_wattr, left, top, right, bottom, unused_maxx, unused_maxy = struct.unpack(b'hhhhHhhhhhh', csbi.raw)
    x = right - left + 1
    y = bottom - top + 1
    return (x, y)