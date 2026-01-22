from pyglet.libs.win32.com import pIUnknown
from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
class GdiplusStartupOutput(Structure):
    _fields = [('NotificationHookProc', c_void_p), ('NotificationUnhookProc', c_void_p)]