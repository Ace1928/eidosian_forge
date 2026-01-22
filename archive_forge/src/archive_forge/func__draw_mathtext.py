import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def _draw_mathtext(self, gc, x, y, s, prop, angle):
    ctx = gc.ctx
    width, height, descent, glyphs, rects = self._text2path.mathtext_parser.parse(s, self.dpi, prop)
    ctx.save()
    ctx.translate(x, y)
    if angle:
        ctx.rotate(np.deg2rad(-angle))
    for font, fontsize, idx, ox, oy in glyphs:
        ctx.new_path()
        ctx.move_to(ox, -oy)
        ctx.select_font_face(*_cairo_font_args_from_font_prop(ttfFontProperty(font)))
        ctx.set_font_size(self.points_to_pixels(fontsize))
        ctx.show_text(chr(idx))
    for ox, oy, w, h in rects:
        ctx.new_path()
        ctx.rectangle(ox, -oy, w, -h)
        ctx.set_source_rgb(0, 0, 0)
        ctx.fill_preserve()
    ctx.restore()