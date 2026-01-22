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
class PSFile:
    """
    Wrapper for bytesio object that treats either CR or LF as end of line.
    This class is no longer used internally, but kept for backwards compatibility.
    """

    def __init__(self, fp):
        deprecate('PSFile', 11, action='If you need the functionality of this class you will need to implement it yourself.')
        self.fp = fp
        self.char = None

    def seek(self, offset, whence=io.SEEK_SET):
        self.char = None
        self.fp.seek(offset, whence)

    def readline(self):
        s = [self.char or b'']
        self.char = None
        c = self.fp.read(1)
        while c not in b'\r\n' and len(c):
            s.append(c)
            c = self.fp.read(1)
        self.char = self.fp.read(1)
        if self.char in b'\r\n':
            self.char = None
        return b''.join(s).decode('latin-1')