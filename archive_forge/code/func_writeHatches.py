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
def writeHatches(self):
    hatchDict = dict()
    sidelen = 72.0
    for hatch_style, name in self.hatchPatterns.items():
        ob = self.reserveObject('hatch pattern')
        hatchDict[name] = ob
        res = {'Procsets': [Name(x) for x in 'PDF Text ImageB ImageC ImageI'.split()]}
        self.beginStream(ob.id, None, {'Type': Name('Pattern'), 'PatternType': 1, 'PaintType': 1, 'TilingType': 1, 'BBox': [0, 0, sidelen, sidelen], 'XStep': sidelen, 'YStep': sidelen, 'Resources': res, 'Matrix': [1, 0, 0, 1, 0, self.height * 72]})
        stroke_rgb, fill_rgb, hatch = hatch_style
        self.output(stroke_rgb[0], stroke_rgb[1], stroke_rgb[2], Op.setrgb_stroke)
        if fill_rgb is not None:
            self.output(fill_rgb[0], fill_rgb[1], fill_rgb[2], Op.setrgb_nonstroke, 0, 0, sidelen, sidelen, Op.rectangle, Op.fill)
        self.output(mpl.rcParams['hatch.linewidth'], Op.setlinewidth)
        self.output(*self.pathOperations(Path.hatch(hatch), Affine2D().scale(sidelen), simplify=False))
        self.output(Op.fill_stroke)
        self.endStream()
    self.writeObject(self.hatchObject, hatchDict)