import re
from base64 import b64decode
import imghdr
from kivy.event import EventDispatcher
from kivy.core import core_register_libs
from kivy.logger import Logger
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.atlas import Atlas
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.compat import string_types
from kivy.setupconfig import USE_SDL2
import zipfile
from io import BytesIO
from os import environ
from kivy.graphics.texture import Texture, TextureRegion
class ImageLoaderBase(object):
    """Base to implement an image loader."""
    __slots__ = ('_texture', '_data', 'filename', 'keep_data', '_mipmap', '_nocache', '_ext', '_inline')

    def __init__(self, filename, **kwargs):
        self._mipmap = kwargs.get('mipmap', False)
        self.keep_data = kwargs.get('keep_data', False)
        self._nocache = kwargs.get('nocache', False)
        self._ext = kwargs.get('ext')
        self._inline = kwargs.get('inline')
        self.filename = filename
        if self._inline:
            self._data = self.load(kwargs.get('rawdata'))
        else:
            self._data = self.load(filename)
        self._textures = None

    def load(self, filename):
        """Load an image"""
        return None

    @staticmethod
    def can_save(fmt, is_bytesio=False):
        """Indicate if the loader can save the Image object

        .. versionchanged:: 1.11.0
            Parameter `fmt` and `is_bytesio` added
        """
        return False

    @staticmethod
    def can_load_memory():
        """Indicate if the loader can load an image by passing data
        """
        return False

    @staticmethod
    def save(*largs, **kwargs):
        raise NotImplementedError()

    def populate(self):
        self._textures = []
        fname = self.filename
        if __debug__:
            Logger.trace('Image: %r, populate to textures (%d)' % (fname, len(self._data)))
        for count in range(len(self._data)):
            chr = type(fname)
            uid = chr(u'%s|%d|%d') % (fname, self._mipmap, count)
            texture = Cache.get('kv.texture', uid)
            if texture is None:
                imagedata = self._data[count]
                source = '{}{}|'.format('zip|' if fname.endswith('.zip') else '', self._nocache)
                imagedata.source = chr(source) + uid
                texture = Texture.create_from_data(imagedata, mipmap=self._mipmap)
                if not self._nocache:
                    Cache.append('kv.texture', uid, texture)
                if imagedata.flip_vertical:
                    texture.flip_vertical()
            self._textures.append(texture)
            if not self.keep_data:
                self._data[count].release_data()

    @property
    def width(self):
        """Image width
        """
        return self._data[0].width

    @property
    def height(self):
        """Image height
        """
        return self._data[0].height

    @property
    def size(self):
        """Image size (width, height)
        """
        return (self._data[0].width, self._data[0].height)

    @property
    def texture(self):
        """Get the image texture (created on the first call)
        """
        if self._textures is None:
            self.populate()
        if self._textures is None:
            return None
        return self._textures[0]

    @property
    def textures(self):
        """Get the textures list (for mipmapped image or animated image)

        .. versionadded:: 1.0.8
        """
        if self._textures is None:
            self.populate()
        return self._textures

    @property
    def nocache(self):
        """Indicate if the texture will not be stored in the cache

        .. versionadded:: 1.6.0
        """
        return self._nocache