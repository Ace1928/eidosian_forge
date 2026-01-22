import ctypes
import warnings
from collections import namedtuple
from pyglet.util import asbytes, asstr
from pyglet.font import base
from pyglet import image
from pyglet.font.fontconfig import get_fontconfig
from pyglet.font.freetype_lib import *
def _load_font_face(self):
    self.face = self._memory_faces.get(self._name, self.bold, self.italic)
    if self.face is None:
        self._load_font_face_from_system()