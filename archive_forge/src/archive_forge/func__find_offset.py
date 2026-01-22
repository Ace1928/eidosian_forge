from __future__ import annotations
import io
import os
import re
import subprocess
import sys
import tempfile
from . import Image, ImageFile
from ._binary import i32le as i32
from ._deprecate import deprecate
def _find_offset(self, fp):
    s = fp.read(4)
    if s == b'%!PS':
        fp.seek(0, io.SEEK_END)
        length = fp.tell()
        offset = 0
    elif i32(s) == 3335770309:
        s = fp.read(8)
        offset = i32(s)
        length = i32(s, 4)
    else:
        msg = 'not an EPS file'
        raise SyntaxError(msg)
    return (length, offset)