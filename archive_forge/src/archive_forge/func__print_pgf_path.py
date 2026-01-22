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
def _print_pgf_path(self, gc, path, transform, rgbFace=None):
    f = 1.0 / self.dpi
    bbox = gc.get_clip_rectangle() if gc else None
    maxcoord = 16383 / 72.27 * self.dpi
    if bbox and rgbFace is None:
        p1, p2 = bbox.get_points()
        clip = (max(p1[0], -maxcoord), max(p1[1], -maxcoord), min(p2[0], maxcoord), min(p2[1], maxcoord))
    else:
        clip = (-maxcoord, -maxcoord, maxcoord, maxcoord)
    for points, code in path.iter_segments(transform, clip=clip):
        if code == Path.MOVETO:
            x, y = tuple(points)
            _writeln(self.fh, '\\pgfpathmoveto{\\pgfqpoint{%fin}{%fin}}' % (f * x, f * y))
        elif code == Path.CLOSEPOLY:
            _writeln(self.fh, '\\pgfpathclose')
        elif code == Path.LINETO:
            x, y = tuple(points)
            _writeln(self.fh, '\\pgfpathlineto{\\pgfqpoint{%fin}{%fin}}' % (f * x, f * y))
        elif code == Path.CURVE3:
            cx, cy, px, py = tuple(points)
            coords = (cx * f, cy * f, px * f, py * f)
            _writeln(self.fh, '\\pgfpathquadraticcurveto{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}' % coords)
        elif code == Path.CURVE4:
            c1x, c1y, c2x, c2y, px, py = tuple(points)
            coords = (c1x * f, c1y * f, c2x * f, c2y * f, px * f, py * f)
            _writeln(self.fh, '\\pgfpathcurveto{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}{\\pgfqpoint{%fin}{%fin}}' % coords)
    sketch_params = gc.get_sketch_params() if gc else None
    if sketch_params is not None:
        scale, length, randomness = sketch_params
        if scale is not None:
            length *= 0.5
            scale *= 2
            _writeln(self.fh, '\\usepgfmodule{decorations}')
            _writeln(self.fh, '\\usepgflibrary{decorations.pathmorphing}')
            _writeln(self.fh, f'\\pgfkeys{{/pgf/decoration/.cd, segment length = {length * f:f}in, amplitude = {scale * f:f}in}}')
            _writeln(self.fh, f'\\pgfmathsetseed{{{int(randomness)}}}')
            _writeln(self.fh, '\\pgfdecoratecurrentpath{random steps}')