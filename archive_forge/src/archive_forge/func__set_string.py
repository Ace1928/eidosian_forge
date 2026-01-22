from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def _set_string(self, name, value):
    assert self._pattern
    assert name
    assert self._fontconfig
    if not value:
        return
    value = value.encode('utf8')
    self._fontconfig.FcPatternAddString(self._pattern, name, asbytes(value))