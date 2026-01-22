import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
class FigureCanvasCairo(FigureCanvasBase):

    @property
    def _renderer(self):
        if not hasattr(self, '_cached_renderer'):
            self._cached_renderer = RendererCairo(self.figure.dpi)
        return self._cached_renderer

    def get_renderer(self):
        return self._renderer

    def copy_from_bbox(self, bbox):
        surface = self._renderer.gc.ctx.get_target()
        if not isinstance(surface, cairo.ImageSurface):
            raise RuntimeError('copy_from_bbox only works when rendering to an ImageSurface')
        sw = surface.get_width()
        sh = surface.get_height()
        x0 = math.ceil(bbox.x0)
        x1 = math.floor(bbox.x1)
        y0 = math.ceil(sh - bbox.y1)
        y1 = math.floor(sh - bbox.y0)
        if not (0 <= x0 and x1 <= sw and (bbox.x0 <= bbox.x1) and (0 <= y0) and (y1 <= sh) and (bbox.y0 <= bbox.y1)):
            raise ValueError('Invalid bbox')
        sls = (slice(y0, y0 + max(y1 - y0, 0)), slice(x0, x0 + max(x1 - x0, 0)))
        data = np.frombuffer(surface.get_data(), np.uint32).reshape((sh, sw))[sls].copy()
        return _CairoRegion(sls, data)

    def restore_region(self, region):
        surface = self._renderer.gc.ctx.get_target()
        if not isinstance(surface, cairo.ImageSurface):
            raise RuntimeError('restore_region only works when rendering to an ImageSurface')
        surface.flush()
        sw = surface.get_width()
        sh = surface.get_height()
        sly, slx = region._slices
        np.frombuffer(surface.get_data(), np.uint32).reshape((sh, sw))[sly, slx] = region._data
        surface.mark_dirty_rectangle(slx.start, sly.start, slx.stop - slx.start, sly.stop - sly.start)

    def print_png(self, fobj):
        self._get_printed_image_surface().write_to_png(fobj)

    def print_rgba(self, fobj):
        width, height = self.get_width_height()
        buf = self._get_printed_image_surface().get_data()
        fobj.write(cbook._premultiplied_argb32_to_unmultiplied_rgba8888(np.asarray(buf).reshape((width, height, 4))))
    print_raw = print_rgba

    def _get_printed_image_surface(self):
        self._renderer.dpi = self.figure.dpi
        width, height = self.get_width_height()
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        self._renderer.set_context(cairo.Context(surface))
        self.figure.draw(self._renderer)
        return surface

    def _save(self, fmt, fobj, *, orientation='portrait'):
        dpi = 72
        self.figure.dpi = dpi
        w_in, h_in = self.figure.get_size_inches()
        width_in_points, height_in_points = (w_in * dpi, h_in * dpi)
        if orientation == 'landscape':
            width_in_points, height_in_points = (height_in_points, width_in_points)
        if fmt == 'ps':
            if not hasattr(cairo, 'PSSurface'):
                raise RuntimeError('cairo has not been compiled with PS support enabled')
            surface = cairo.PSSurface(fobj, width_in_points, height_in_points)
        elif fmt == 'pdf':
            if not hasattr(cairo, 'PDFSurface'):
                raise RuntimeError('cairo has not been compiled with PDF support enabled')
            surface = cairo.PDFSurface(fobj, width_in_points, height_in_points)
        elif fmt in ('svg', 'svgz'):
            if not hasattr(cairo, 'SVGSurface'):
                raise RuntimeError('cairo has not been compiled with SVG support enabled')
            if fmt == 'svgz':
                if isinstance(fobj, str):
                    fobj = gzip.GzipFile(fobj, 'wb')
                else:
                    fobj = gzip.GzipFile(None, 'wb', fileobj=fobj)
            surface = cairo.SVGSurface(fobj, width_in_points, height_in_points)
        else:
            raise ValueError(f'Unknown format: {fmt!r}')
        self._renderer.dpi = self.figure.dpi
        self._renderer.set_context(cairo.Context(surface))
        ctx = self._renderer.gc.ctx
        if orientation == 'landscape':
            ctx.rotate(np.pi / 2)
            ctx.translate(0, -height_in_points)
        self.figure.draw(self._renderer)
        ctx.show_page()
        surface.finish()
        if fmt == 'svgz':
            fobj.close()
    print_pdf = functools.partialmethod(_save, 'pdf')
    print_ps = functools.partialmethod(_save, 'ps')
    print_svg = functools.partialmethod(_save, 'svg')
    print_svgz = functools.partialmethod(_save, 'svgz')