import codecs
import datetime
from enum import Enum
import functools
from io import StringIO
import itertools
import logging
import os
import pathlib
import shutil
from tempfile import TemporaryDirectory
import time
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _path, _text_helpers
from matplotlib._afm import AFM
from matplotlib.backend_bases import (
from matplotlib.cbook import is_writable_file_like, file_requires_unicode
from matplotlib.font_manager import get_font
from matplotlib.ft2font import LOAD_NO_SCALE, FT2Font
from matplotlib._ttconv import convert_ttf_to_ps
from matplotlib._mathtext_data import uni2type1
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib.backends.backend_mixed import MixedModeRenderer
from . import _backend_pdf_ps
def _draw_ps(self, ps, gc, rgbFace, *, fill=True, stroke=True):
    """
        Emit the PostScript snippet *ps* with all the attributes from *gc*
        applied.  *ps* must consist of PostScript commands to construct a path.

        The *fill* and/or *stroke* kwargs can be set to False if the *ps*
        string already includes filling and/or stroking, in which case
        `_draw_ps` is just supplying properties and clipping.
        """
    write = self._pswriter.write
    mightstroke = gc.get_linewidth() > 0 and (not self._is_transparent(gc.get_rgb()))
    if not mightstroke:
        stroke = False
    if self._is_transparent(rgbFace):
        fill = False
    hatch = gc.get_hatch()
    if mightstroke:
        self.set_linewidth(gc.get_linewidth())
        self.set_linejoin(gc.get_joinstyle())
        self.set_linecap(gc.get_capstyle())
        self.set_linedash(*gc.get_dashes())
    if mightstroke or hatch:
        self.set_color(*gc.get_rgb()[:3])
    write('gsave\n')
    write(self._get_clip_cmd(gc))
    write(ps.strip())
    write('\n')
    if fill:
        if stroke or hatch:
            write('gsave\n')
        self.set_color(*rgbFace[:3], store=False)
        write('fill\n')
        if stroke or hatch:
            write('grestore\n')
    if hatch:
        hatch_name = self.create_hatch(hatch)
        write('gsave\n')
        write(_nums_to_str(*gc.get_hatch_color()[:3]))
        write(f' {hatch_name} setpattern fill grestore\n')
    if stroke:
        write('stroke\n')
    write('grestore\n')