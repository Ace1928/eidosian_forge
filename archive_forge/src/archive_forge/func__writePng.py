import codecs
from datetime import timezone
from datetime import datetime
from enum import Enum
from functools import total_ordering
from io import BytesIO
import itertools
import logging
import math
import os
import string
import struct
import sys
import time
import types
import warnings
import zlib
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, _text_helpers, _type1font, cbook, dviread
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.figure import Figure
from matplotlib.font_manager import get_font, fontManager as _fontManager
from matplotlib._afm import AFM
from matplotlib.ft2font import (FIXED_WIDTH, ITALIC, LOAD_NO_SCALE,
from matplotlib.transforms import Affine2D, BboxBase
from matplotlib.path import Path
from matplotlib.dates import UTC
from matplotlib import _path
from . import _backend_pdf_ps
def _writePng(self, img):
    """
        Write the image *img* into the pdf file using png
        predictors with Flate compression.
        """
    buffer = BytesIO()
    img.save(buffer, format='png')
    buffer.seek(8)
    png_data = b''
    bit_depth = palette = None
    while True:
        length, type = struct.unpack(b'!L4s', buffer.read(8))
        if type in [b'IHDR', b'PLTE', b'IDAT']:
            data = buffer.read(length)
            if len(data) != length:
                raise RuntimeError('truncated data')
            if type == b'IHDR':
                bit_depth = int(data[8])
            elif type == b'PLTE':
                palette = data
            elif type == b'IDAT':
                png_data += data
        elif type == b'IEND':
            break
        else:
            buffer.seek(length, 1)
        buffer.seek(4, 1)
    return (png_data, bit_depth, palette)