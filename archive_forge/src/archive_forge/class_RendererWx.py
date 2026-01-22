import functools
import logging
import math
import pathlib
import sys
import weakref
import numpy as np
import PIL.Image
import matplotlib as mpl
from matplotlib.backend_bases import (
from matplotlib import _api, cbook, backend_tools
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import wx
@_api.deprecated('2.0', name='wx', obj_type='backend', removal='the future', alternative='wxagg', addendum='See the Matplotlib usage FAQ for more info on backends.')
class RendererWx(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles. It acts as the
    'renderer' instance used by many classes in the hierarchy.
    """
    fontweights = {100: wx.FONTWEIGHT_LIGHT, 200: wx.FONTWEIGHT_LIGHT, 300: wx.FONTWEIGHT_LIGHT, 400: wx.FONTWEIGHT_NORMAL, 500: wx.FONTWEIGHT_NORMAL, 600: wx.FONTWEIGHT_NORMAL, 700: wx.FONTWEIGHT_BOLD, 800: wx.FONTWEIGHT_BOLD, 900: wx.FONTWEIGHT_BOLD, 'ultralight': wx.FONTWEIGHT_LIGHT, 'light': wx.FONTWEIGHT_LIGHT, 'normal': wx.FONTWEIGHT_NORMAL, 'medium': wx.FONTWEIGHT_NORMAL, 'semibold': wx.FONTWEIGHT_NORMAL, 'bold': wx.FONTWEIGHT_BOLD, 'heavy': wx.FONTWEIGHT_BOLD, 'ultrabold': wx.FONTWEIGHT_BOLD, 'black': wx.FONTWEIGHT_BOLD}
    fontangles = {'italic': wx.FONTSTYLE_ITALIC, 'normal': wx.FONTSTYLE_NORMAL, 'oblique': wx.FONTSTYLE_SLANT}
    fontnames = {'Sans': wx.FONTFAMILY_SWISS, 'Roman': wx.FONTFAMILY_ROMAN, 'Script': wx.FONTFAMILY_SCRIPT, 'Decorative': wx.FONTFAMILY_DECORATIVE, 'Modern': wx.FONTFAMILY_MODERN, 'Courier': wx.FONTFAMILY_MODERN, 'courier': wx.FONTFAMILY_MODERN}

    def __init__(self, bitmap, dpi):
        """Initialise a wxWindows renderer instance."""
        super().__init__()
        _log.debug('%s - __init__()', type(self))
        self.width = bitmap.GetWidth()
        self.height = bitmap.GetHeight()
        self.bitmap = bitmap
        self.fontd = {}
        self.dpi = dpi
        self.gc = None

    def flipy(self):
        return True

    def get_text_width_height_descent(self, s, prop, ismath):
        if ismath:
            s = cbook.strip_math(s)
        if self.gc is None:
            gc = self.new_gc()
        else:
            gc = self.gc
        gfx_ctx = gc.gfx_ctx
        font = self.get_wx_font(s, prop)
        gfx_ctx.SetFont(font, wx.BLACK)
        w, h, descent, leading = gfx_ctx.GetFullTextExtent(s)
        return (w, h, descent)

    def get_canvas_width_height(self):
        return (self.width, self.height)

    def handle_clip_rectangle(self, gc):
        new_bounds = gc.get_clip_rectangle()
        if new_bounds is not None:
            new_bounds = new_bounds.bounds
        gfx_ctx = gc.gfx_ctx
        if gfx_ctx._lastcliprect != new_bounds:
            gfx_ctx._lastcliprect = new_bounds
            if new_bounds is None:
                gfx_ctx.ResetClip()
            else:
                gfx_ctx.Clip(new_bounds[0], self.height - new_bounds[1] - new_bounds[3], new_bounds[2], new_bounds[3])

    @staticmethod
    def convert_path(gfx_ctx, path, transform):
        wxpath = gfx_ctx.CreatePath()
        for points, code in path.iter_segments(transform):
            if code == Path.MOVETO:
                wxpath.MoveToPoint(*points)
            elif code == Path.LINETO:
                wxpath.AddLineToPoint(*points)
            elif code == Path.CURVE3:
                wxpath.AddQuadCurveToPoint(*points)
            elif code == Path.CURVE4:
                wxpath.AddCurveToPoint(*points)
            elif code == Path.CLOSEPOLY:
                wxpath.CloseSubpath()
        return wxpath

    def draw_path(self, gc, path, transform, rgbFace=None):
        gc.select()
        self.handle_clip_rectangle(gc)
        gfx_ctx = gc.gfx_ctx
        transform = transform + Affine2D().scale(1.0, -1.0).translate(0.0, self.height)
        wxpath = self.convert_path(gfx_ctx, path, transform)
        if rgbFace is not None:
            gfx_ctx.SetBrush(wx.Brush(gc.get_wxcolour(rgbFace)))
            gfx_ctx.DrawPath(wxpath)
        else:
            gfx_ctx.StrokePath(wxpath)
        gc.unselect()

    def draw_image(self, gc, x, y, im):
        bbox = gc.get_clip_rectangle()
        if bbox is not None:
            l, b, w, h = bbox.bounds
        else:
            l = 0
            b = 0
            w = self.width
            h = self.height
        rows, cols = im.shape[:2]
        bitmap = wx.Bitmap.FromBufferRGBA(cols, rows, im.tobytes())
        gc.select()
        gc.gfx_ctx.DrawBitmap(bitmap, int(l), int(self.height - b), int(w), int(-h))
        gc.unselect()

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if ismath:
            s = cbook.strip_math(s)
        _log.debug('%s - draw_text()', type(self))
        gc.select()
        self.handle_clip_rectangle(gc)
        gfx_ctx = gc.gfx_ctx
        font = self.get_wx_font(s, prop)
        color = gc.get_wxcolour(gc.get_rgb())
        gfx_ctx.SetFont(font, color)
        w, h, d = self.get_text_width_height_descent(s, prop, ismath)
        x = int(x)
        y = int(y - h)
        if angle == 0.0:
            gfx_ctx.DrawText(s, x, y)
        else:
            rads = math.radians(angle)
            xo = h * math.sin(rads)
            yo = h * math.cos(rads)
            gfx_ctx.DrawRotatedText(s, x - xo, y - yo, rads)
        gc.unselect()

    def new_gc(self):
        _log.debug('%s - new_gc()', type(self))
        self.gc = GraphicsContextWx(self.bitmap, self)
        self.gc.select()
        self.gc.unselect()
        return self.gc

    def get_wx_font(self, s, prop):
        """Return a wx font.  Cache font instances for efficiency."""
        _log.debug('%s - get_wx_font()', type(self))
        key = hash(prop)
        font = self.fontd.get(key)
        if font is not None:
            return font
        size = self.points_to_pixels(prop.get_size_in_points())
        self.fontd[key] = font = wx.Font(pointSize=round(size), family=self.fontnames.get(prop.get_name(), wx.ROMAN), style=self.fontangles[prop.get_style()], weight=self.fontweights[prop.get_weight()])
        return font

    def points_to_pixels(self, points):
        return points * (PIXELS_PER_INCH / 72.0 * self.dpi / 72.0)