import os
import zlib
import logging
from io import BytesIO
import numpy as np
from ..core import Format, read_n_bytes, image_as_uint
def _complete(self):
    if not self._framecounter:
        self._write_header((10, 10), self._arg_fps)
    if not self._arg_loop:
        self._fp.write(_swf.DoActionTag('stop').get_tag())
    self._fp.write('\x00\x00'.encode('ascii'))