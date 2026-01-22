from collections import OrderedDict
from ctypes import *
import pyglet.lib
from pyglet.util import asbytes, asstr
from pyglet.font.base import FontException
def _substitute_defaults(self):
    assert self._pattern
    assert self._fontconfig
    self._fontconfig.FcConfigSubstitute(None, self._pattern, FcMatchPattern)
    self._fontconfig.FcDefaultSubstitute(self._pattern)