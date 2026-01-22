from contextlib import nullcontext
from math import radians, cos, sin
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.backend_bases import (
from matplotlib.font_manager import fontManager as _fontManager, get_font
from matplotlib.ft2font import (LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING,
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxBase
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg
class RendererAgg(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles
    """

    def __init__(self, width, height, dpi):
        super().__init__()
        self.dpi = dpi
        self.width = width
        self.height = height
        self._renderer = _RendererAgg(int(width), int(height), dpi)
        self._filter_renderers = []
        self._update_methods()
        self.mathtext_parser = MathTextParser('agg')
        self.bbox = Bbox.from_bounds(0, 0, self.width, self.height)

    def __getstate__(self):
        return {'width': self.width, 'height': self.height, 'dpi': self.dpi}

    def __setstate__(self, state):
        self.__init__(state['width'], state['height'], state['dpi'])

    def _update_methods(self):
        self.draw_gouraud_triangle = self._renderer.draw_gouraud_triangle
        self.draw_gouraud_triangles = self._renderer.draw_gouraud_triangles
        self.draw_image = self._renderer.draw_image
        self.draw_markers = self._renderer.draw_markers
        self.draw_path_collection = self._renderer.draw_path_collection
        self.draw_quad_mesh = self._renderer.draw_quad_mesh
        self.copy_from_bbox = self._renderer.copy_from_bbox

    def draw_path(self, gc, path, transform, rgbFace=None):
        nmax = mpl.rcParams['agg.path.chunksize']
        npts = path.vertices.shape[0]
        if npts > nmax > 100 and path.should_simplify and (rgbFace is None) and (gc.get_hatch() is None):
            nch = np.ceil(npts / nmax)
            chsize = int(np.ceil(npts / nch))
            i0 = np.arange(0, npts, chsize)
            i1 = np.zeros_like(i0)
            i1[:-1] = i0[1:] - 1
            i1[-1] = npts
            for ii0, ii1 in zip(i0, i1):
                v = path.vertices[ii0:ii1, :]
                c = path.codes
                if c is not None:
                    c = c[ii0:ii1]
                    c[0] = Path.MOVETO
                p = Path(v, c)
                p.simplify_threshold = path.simplify_threshold
                try:
                    self._renderer.draw_path(gc, p, transform, rgbFace)
                except OverflowError:
                    msg = f"Exceeded cell block limit in Agg.\n\nPlease reduce the value of rcParams['agg.path.chunksize'] (currently {nmax}) or increase the path simplification threshold(rcParams['path.simplify_threshold'] = {mpl.rcParams['path.simplify_threshold']:.2f} by default and path.simplify_threshold = {path.simplify_threshold:.2f} on the input)."
                    raise OverflowError(msg) from None
        else:
            try:
                self._renderer.draw_path(gc, path, transform, rgbFace)
            except OverflowError:
                cant_chunk = ''
                if rgbFace is not None:
                    cant_chunk += '- cannot split filled path\n'
                if gc.get_hatch() is not None:
                    cant_chunk += '- cannot split hatched path\n'
                if not path.should_simplify:
                    cant_chunk += '- path.should_simplify is False\n'
                if len(cant_chunk):
                    msg = f'Exceeded cell block limit in Agg, however for the following reasons:\n\n{cant_chunk}\nwe cannot automatically split up this path to draw.\n\nPlease manually simplify your path.'
                else:
                    inc_threshold = f"or increase the path simplification threshold(rcParams['path.simplify_threshold'] = {mpl.rcParams['path.simplify_threshold']} by default and path.simplify_threshold = {path.simplify_threshold} on the input)."
                    if nmax > 100:
                        msg = f"Exceeded cell block limit in Agg.  Please reduce the value of rcParams['agg.path.chunksize'] (currently {nmax}) {inc_threshold}"
                    else:
                        msg = f"Exceeded cell block limit in Agg.  Please set the value of rcParams['agg.path.chunksize'], (currently {nmax}) to be greater than 100 " + inc_threshold
                raise OverflowError(msg) from None

    def draw_mathtext(self, gc, x, y, s, prop, angle):
        """Draw mathtext using :mod:`matplotlib.mathtext`."""
        ox, oy, width, height, descent, font_image = self.mathtext_parser.parse(s, self.dpi, prop, antialiased=gc.get_antialiased())
        xd = descent * sin(radians(angle))
        yd = descent * cos(radians(angle))
        x = round(x + ox + xd)
        y = round(y - oy + yd)
        self._renderer.draw_text_image(font_image, x, y + 1, angle, gc)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)
        font = self._prepare_font(prop)
        font.set_text(s, 0, flags=get_hinting_flag())
        font.draw_glyphs_to_bitmap(antialiased=gc.get_antialiased())
        d = font.get_descent() / 64.0
        xo, yo = font.get_bitmap_offset()
        xo /= 64.0
        yo /= 64.0
        xd = d * sin(radians(angle))
        yd = d * cos(radians(angle))
        x = round(x + xo + xd)
        y = round(y + yo + yd)
        self._renderer.draw_text_image(font, x, y + 1, angle, gc)

    def get_text_width_height_descent(self, s, prop, ismath):
        _api.check_in_list(['TeX', True, False], ismath=ismath)
        if ismath == 'TeX':
            return super().get_text_width_height_descent(s, prop, ismath)
        if ismath:
            ox, oy, width, height, descent, font_image = self.mathtext_parser.parse(s, self.dpi, prop)
            return (width, height, descent)
        font = self._prepare_font(prop)
        font.set_text(s, 0.0, flags=get_hinting_flag())
        w, h = font.get_width_height()
        d = font.get_descent()
        w /= 64.0
        h /= 64.0
        d /= 64.0
        return (w, h, d)

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        size = prop.get_size_in_points()
        texmanager = self.get_texmanager()
        Z = texmanager.get_grey(s, size, self.dpi)
        Z = np.array(Z * 255.0, np.uint8)
        w, h, d = self.get_text_width_height_descent(s, prop, ismath='TeX')
        xd = d * sin(radians(angle))
        yd = d * cos(radians(angle))
        x = round(x + xd)
        y = round(y + yd)
        self._renderer.draw_text_image(Z, x, y, angle, gc)

    def get_canvas_width_height(self):
        return (self.width, self.height)

    def _prepare_font(self, font_prop):
        """
        Get the `.FT2Font` for *font_prop*, clear its buffer, and set its size.
        """
        font = get_font(_fontManager._find_fonts_by_props(font_prop))
        font.clear()
        size = font_prop.get_size_in_points()
        font.set_size(size, self.dpi)
        return font

    def points_to_pixels(self, points):
        return points * self.dpi / 72

    def buffer_rgba(self):
        return memoryview(self._renderer)

    def tostring_argb(self):
        return np.asarray(self._renderer).take([3, 0, 1, 2], axis=2).tobytes()

    @_api.deprecated('3.8', alternative='buffer_rgba')
    def tostring_rgb(self):
        return np.asarray(self._renderer).take([0, 1, 2], axis=2).tobytes()

    def clear(self):
        self._renderer.clear()

    def option_image_nocomposite(self):
        return True

    def option_scale_image(self):
        return False

    def restore_region(self, region, bbox=None, xy=None):
        """
        Restore the saved region. If bbox (instance of BboxBase, or
        its extents) is given, only the region specified by the bbox
        will be restored. *xy* (a pair of floats) optionally
        specifies the new position (the LLC of the original region,
        not the LLC of the bbox) where the region will be restored.

        >>> region = renderer.copy_from_bbox()
        >>> x1, y1, x2, y2 = region.get_extents()
        >>> renderer.restore_region(region, bbox=(x1+dx, y1, x2, y2),
        ...                         xy=(x1-dx, y1))

        """
        if bbox is not None or xy is not None:
            if bbox is None:
                x1, y1, x2, y2 = region.get_extents()
            elif isinstance(bbox, BboxBase):
                x1, y1, x2, y2 = bbox.extents
            else:
                x1, y1, x2, y2 = bbox
            if xy is None:
                ox, oy = (x1, y1)
            else:
                ox, oy = xy
            self._renderer.restore_region(region, int(x1), int(y1), int(x2), int(y2), int(ox), int(oy))
        else:
            self._renderer.restore_region(region)

    def start_filter(self):
        """
        Start filtering. It simply creates a new canvas (the old one is saved).
        """
        self._filter_renderers.append(self._renderer)
        self._renderer = _RendererAgg(int(self.width), int(self.height), self.dpi)
        self._update_methods()

    def stop_filter(self, post_processing):
        """
        Save the current canvas as an image and apply post processing.

        The *post_processing* function::

           def post_processing(image, dpi):
             # ny, nx, depth = image.shape
             # image (numpy array) has RGBA channels and has a depth of 4.
             ...
             # create a new_image (numpy array of 4 channels, size can be
             # different). The resulting image may have offsets from
             # lower-left corner of the original image
             return new_image, offset_x, offset_y

        The saved renderer is restored and the returned image from
        post_processing is plotted (using draw_image) on it.
        """
        orig_img = np.asarray(self.buffer_rgba())
        slice_y, slice_x = cbook._get_nonzero_slices(orig_img[..., 3])
        cropped_img = orig_img[slice_y, slice_x]
        self._renderer = self._filter_renderers.pop()
        self._update_methods()
        if cropped_img.size:
            img, ox, oy = post_processing(cropped_img / 255, self.dpi)
            gc = self.new_gc()
            if img.dtype.kind == 'f':
                img = np.asarray(img * 255.0, np.uint8)
            self._renderer.draw_image(gc, slice_x.start + ox, int(self.height) - slice_y.stop + oy, img[::-1])