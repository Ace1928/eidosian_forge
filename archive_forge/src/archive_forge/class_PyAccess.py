from __future__ import annotations
import logging
import sys
from ._deprecate import deprecate
class PyAccess:

    def __init__(self, img, readonly=False):
        deprecate('PyAccess', 11)
        vals = dict(img.im.unsafe_ptrs)
        self.readonly = readonly
        self.image8 = ffi.cast('unsigned char **', vals['image8'])
        self.image32 = ffi.cast('int **', vals['image32'])
        self.image = ffi.cast('unsigned char **', vals['image'])
        self.xsize, self.ysize = img.im.size
        self._img = img
        self._im = img.im
        if self._im.mode in ('P', 'PA'):
            self._palette = img.palette
        self._post_init()

    def _post_init(self):
        pass

    def __setitem__(self, xy, color):
        """
        Modifies the pixel at x,y. The color is given as a single
        numerical value for single band images, and a tuple for
        multi-band images

        :param xy: The pixel coordinate, given as (x, y). See
           :ref:`coordinate-system`.
        :param color: The pixel value.
        """
        if self.readonly:
            msg = 'Attempt to putpixel a read only image'
            raise ValueError(msg)
        x, y = xy
        if x < 0:
            x = self.xsize + x
        if y < 0:
            y = self.ysize + y
        x, y = self.check_xy((x, y))
        if self._im.mode in ('P', 'PA') and isinstance(color, (list, tuple)) and (len(color) in [3, 4]):
            if self._im.mode == 'PA':
                alpha = color[3] if len(color) == 4 else 255
                color = color[:3]
            color = self._palette.getcolor(color, self._img)
            if self._im.mode == 'PA':
                color = (color, alpha)
        return self.set_pixel(x, y, color)

    def __getitem__(self, xy):
        """
        Returns the pixel at x,y. The pixel is returned as a single
        value for single band images or a tuple for multiple band
        images

        :param xy: The pixel coordinate, given as (x, y). See
          :ref:`coordinate-system`.
        :returns: a pixel value for single band images, a tuple of
          pixel values for multiband images.
        """
        x, y = xy
        if x < 0:
            x = self.xsize + x
        if y < 0:
            y = self.ysize + y
        x, y = self.check_xy((x, y))
        return self.get_pixel(x, y)
    putpixel = __setitem__
    getpixel = __getitem__

    def check_xy(self, xy):
        x, y = xy
        if not (0 <= x < self.xsize and 0 <= y < self.ysize):
            msg = 'pixel location out of range'
            raise ValueError(msg)
        return xy