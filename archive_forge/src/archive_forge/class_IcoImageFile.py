from __future__ import annotations
import warnings
from io import BytesIO
from math import ceil, log
from . import BmpImagePlugin, Image, ImageFile, PngImagePlugin
from ._binary import i16le as i16
from ._binary import i32le as i32
from ._binary import o8
from ._binary import o16le as o16
from ._binary import o32le as o32
class IcoImageFile(ImageFile.ImageFile):
    """
    PIL read-only image support for Microsoft Windows .ico files.

    By default the largest resolution image in the file will be loaded. This
    can be changed by altering the 'size' attribute before calling 'load'.

    The info dictionary has a key 'sizes' that is a list of the sizes available
    in the icon file.

    Handles classic, XP and Vista icon formats.

    When saving, PNG compression is used. Support for this was only added in
    Windows Vista. If you are unable to view the icon in Windows, convert the
    image to "RGBA" mode before saving.

    This plugin is a refactored version of Win32IconImagePlugin by Bryan Davis
    <casadebender@gmail.com>.
    https://code.google.com/archive/p/casadebender/wikis/Win32IconImagePlugin.wiki
    """
    format = 'ICO'
    format_description = 'Windows Icon'

    def _open(self):
        self.ico = IcoFile(self.fp)
        self.info['sizes'] = self.ico.sizes()
        self.size = self.ico.entry[0]['dim']
        self.load()

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        if value not in self.info['sizes']:
            msg = 'This is not one of the allowed sizes of this image'
            raise ValueError(msg)
        self._size = value

    def load(self):
        if self.im is not None and self.im.size == self.size:
            return Image.Image.load(self)
        im = self.ico.getimage(self.size)
        im.load()
        self.im = im.im
        self.pyaccess = None
        self._mode = im.mode
        if im.size != self.size:
            warnings.warn('Image was not the expected size')
            index = self.ico.getentryindex(self.size)
            sizes = list(self.info['sizes'])
            sizes[index] = im.size
            self.info['sizes'] = set(sizes)
            self.size = im.size

    def load_seek(self):
        pass