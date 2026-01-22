from __future__ import annotations
import io
import os
import struct
import sys
from . import Image, ImageFile, PngImagePlugin, features
class IcnsImageFile(ImageFile.ImageFile):
    """
    PIL image support for Mac OS .icns files.
    Chooses the best resolution, but will possibly load
    a different size image if you mutate the size attribute
    before calling 'load'.

    The info dictionary has a key 'sizes' that is a list
    of sizes that the icns file has.
    """
    format = 'ICNS'
    format_description = 'Mac OS icns resource'

    def _open(self):
        self.icns = IcnsFile(self.fp)
        self._mode = 'RGBA'
        self.info['sizes'] = self.icns.itersizes()
        self.best_size = self.icns.bestsize()
        self.size = (self.best_size[0] * self.best_size[2], self.best_size[1] * self.best_size[2])

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        info_size = value
        if info_size not in self.info['sizes'] and len(info_size) == 2:
            info_size = (info_size[0], info_size[1], 1)
        if info_size not in self.info['sizes'] and len(info_size) == 3 and (info_size[2] == 1):
            simple_sizes = [(size[0] * size[2], size[1] * size[2]) for size in self.info['sizes']]
            if value in simple_sizes:
                info_size = self.info['sizes'][simple_sizes.index(value)]
        if info_size not in self.info['sizes']:
            msg = 'This is not one of the allowed sizes of this image'
            raise ValueError(msg)
        self._size = value

    def load(self):
        if len(self.size) == 3:
            self.best_size = self.size
            self.size = (self.best_size[0] * self.best_size[2], self.best_size[1] * self.best_size[2])
        px = Image.Image.load(self)
        if self.im is not None and self.im.size == self.size:
            return px
        self.load_prepare()
        im = self.icns.getimage(self.best_size)
        px = im.load()
        self.im = im.im
        self._mode = im.mode
        self.size = im.size
        return px