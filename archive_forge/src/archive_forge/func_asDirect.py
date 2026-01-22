import collections
import io   # For io.BytesIO
import itertools
import math
import operator
import re
import struct
import sys
import warnings
import zlib
from array import array
fromarray = from_array
def asDirect(self):
    """
        Returns the image data as a direct representation of
        an ``x * y * planes`` array.
        This removes the need for callers to deal with
        palettes and transparency themselves.
        Images with a palette (colour type 3) are converted to RGB or RGBA;
        images with transparency (a ``tRNS`` chunk) are converted to
        LA or RGBA as appropriate.
        When returned in this format the pixel values represent
        the colour value directly without needing to refer
        to palettes or transparency information.

        Like the :meth:`read` method this method returns a 4-tuple:

        (*width*, *height*, *rows*, *info*)

        This method normally returns pixel values with
        the bit depth they have in the source image, but
        when the source PNG has an ``sBIT`` chunk it is inspected and
        can reduce the bit depth of the result pixels;
        pixel values will be reduced according to the bit depth
        specified in the ``sBIT`` chunk.
        PNG nerds should note a single result bit depth is
        used for all channels:
        the maximum of the ones specified in the ``sBIT`` chunk.
        An RGB565 image will be rescaled to 6-bit RGB666.

        The *info* dictionary that is returned reflects
        the `direct` format and not the original source image.
        For example, an RGB source image with a ``tRNS`` chunk
        to represent a transparent colour,
        will start with ``planes=3`` and ``alpha=False`` for the
        source image,
        but the *info* dictionary returned by this method
        will have ``planes=4`` and ``alpha=True`` because
        an alpha channel is synthesized and added.

        *rows* is a sequence of rows;
        each row being a sequence of values
        (like the :meth:`read` method).

        All the other aspects of the image data are not changed.
        """
    self.preamble()
    if not self.colormap and (not self.trns) and (not self.sbit):
        return self.read()
    x, y, pixels, info = self.read()
    if self.colormap:
        info['colormap'] = False
        info['alpha'] = bool(self.trns)
        info['bitdepth'] = 8
        info['planes'] = 3 + bool(self.trns)
        plte = self.palette()

        def iterpal(pixels):
            for row in pixels:
                row = [plte[x] for x in row]
                yield array('B', itertools.chain(*row))
        pixels = iterpal(pixels)
    elif self.trns:
        it = self.transparent
        maxval = 2 ** info['bitdepth'] - 1
        planes = info['planes']
        info['alpha'] = True
        info['planes'] += 1
        typecode = 'BH'[info['bitdepth'] > 8]

        def itertrns(pixels):
            for row in pixels:
                row = group(row, planes)
                opa = map(it.__ne__, row)
                opa = map(maxval.__mul__, opa)
                opa = list(zip(opa))
                yield array(typecode, itertools.chain(*map(operator.add, row, opa)))
        pixels = itertrns(pixels)
    targetbitdepth = None
    if self.sbit:
        sbit = struct.unpack(f'{len(self.sbit)}B', self.sbit)
        targetbitdepth = max(sbit)
        if targetbitdepth > info['bitdepth']:
            raise Error(f'sBIT chunk {sbit!r} exceeds bitdepth {self.bitdepth}')
        if min(sbit) <= 0:
            raise Error(f'sBIT chunk {sbit} has a 0-entry')
    if targetbitdepth:
        shift = info['bitdepth'] - targetbitdepth
        info['bitdepth'] = targetbitdepth

        def itershift(pixels):
            for row in pixels:
                yield [p >> shift for p in row]
        pixels = itershift(pixels)
    return (x, y, pixels, info)