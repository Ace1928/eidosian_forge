import codecs
import datetime
import functools
from io import BytesIO
import logging
import math
import os
import pathlib
import shutil
import subprocess
from tempfile import TemporaryDirectory
import weakref
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import (
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib._pylab_helpers import Gcf
def _print_pgf_clip(self, gc):
    f = 1.0 / self.dpi
    bbox = gc.get_clip_rectangle()
    if bbox:
        p1, p2 = bbox.get_points()
        w, h = p2 - p1
        coords = (p1[0] * f, p1[1] * f, w * f, h * f)
        _writeln(self.fh, '\\pgfpathrectangle{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}' % coords)
        _writeln(self.fh, '\\pgfusepath{clip}')
    clippath, clippath_trans = gc.get_clip_path()
    if clippath is not None:
        self._print_pgf_path(gc, clippath, clippath_trans)
        _writeln(self.fh, '\\pgfusepath{clip}')