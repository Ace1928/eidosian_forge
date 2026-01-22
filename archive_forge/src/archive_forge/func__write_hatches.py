import base64
import codecs
import datetime
import gzip
import hashlib
from io import BytesIO
import itertools
import logging
import os
import re
import uuid
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.colors import rgb2hex
from matplotlib.dates import UTC
from matplotlib.path import Path
from matplotlib import _path
from matplotlib.transforms import Affine2D, Affine2DBase
def _write_hatches(self):
    if not len(self._hatchd):
        return
    HATCH_SIZE = 72
    writer = self.writer
    writer.start('defs')
    for (path, face, stroke), oid in self._hatchd.values():
        writer.start('pattern', id=oid, patternUnits='userSpaceOnUse', x='0', y='0', width=str(HATCH_SIZE), height=str(HATCH_SIZE))
        path_data = self._convert_path(path, Affine2D().scale(HATCH_SIZE).scale(1.0, -1.0).translate(0, HATCH_SIZE), simplify=False)
        if face is None:
            fill = 'none'
        else:
            fill = rgb2hex(face)
        writer.element('rect', x='0', y='0', width=str(HATCH_SIZE + 1), height=str(HATCH_SIZE + 1), fill=fill)
        hatch_style = {'fill': rgb2hex(stroke), 'stroke': rgb2hex(stroke), 'stroke-width': str(mpl.rcParams['hatch.linewidth']), 'stroke-linecap': 'butt', 'stroke-linejoin': 'miter'}
        if stroke[3] < 1:
            hatch_style['stroke-opacity'] = str(stroke[3])
        writer.element('path', d=path_data, style=_generate_css(hatch_style))
        writer.end('pattern')
    writer.end('defs')