from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IPropertyBag2(com.pIUnknown):
    _methods_ = [('Read', com.STDMETHOD()), ('Write', com.STDMETHOD()), ('CountProperties', com.STDMETHOD()), ('GetPropertyInfo', com.STDMETHOD()), ('LoadObject', com.STDMETHOD())]