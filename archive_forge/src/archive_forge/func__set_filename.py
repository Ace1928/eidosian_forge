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
def _set_filename(self, value):
    if value is None or value == self._filename:
        return
    self._filename = value
    f = self.filename
    uid = type(f)(u'%s|%d|%d') % (f, self._mipmap, 0)
    image = Cache.get('kv.image', uid)
    if image:
        self.image = image
        if image.__class__ != self.__class__ and (not image.keep_data) and self._keep_data:
            self.remove_from_cache()
            self._filename = ''
            self._set_filename(value)
        else:
            self._texture = None
        return
    else:
        _texture = Cache.get('kv.texture', uid)
        if _texture:
            self._texture = _texture
            return
    tmpfilename = self._filename
    image = ImageLoader.load(self._filename, keep_data=self._keep_data, mipmap=self._mipmap, nocache=self._nocache)
    self._filename = tmpfilename
    if isinstance(image, Texture):
        self._texture = image
        self._size = image.size
    else:
        self.image = image
        if not self._nocache:
            Cache.append('kv.image', uid, self.image)