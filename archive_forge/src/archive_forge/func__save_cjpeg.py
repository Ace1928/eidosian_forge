from __future__ import annotations
import array
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from .JpegPresets import presets
def _save_cjpeg(im, fp, filename):
    tempfile = im._dump()
    subprocess.check_call(['cjpeg', '-outfile', filename, tempfile])
    try:
        os.unlink(tempfile)
    except OSError:
        pass