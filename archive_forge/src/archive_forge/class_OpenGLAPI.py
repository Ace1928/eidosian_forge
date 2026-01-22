import weakref
from enum import Enum
import threading
from typing import Tuple
import pyglet
from pyglet import gl
from pyglet.gl import gl_info
class OpenGLAPI(Enum):
    OPENGL = 1
    OPENGL_ES = 2