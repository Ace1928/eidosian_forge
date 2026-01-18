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
def writeXref(self):
    """Write out the xref table."""
    self.startxref = self.fh.tell() - self.tell_base
    self.write(b'xref\n0 %d\n' % len(self.xrefTable))
    for i, (offset, generation, name) in enumerate(self.xrefTable):
        if offset is None:
            raise AssertionError('No offset for object %d (%s)' % (i, name))
        else:
            key = b'f' if name == 'the zero object' else b'n'
            text = b'%010d %05d %b \n' % (offset, generation, key)
            self.write(text)