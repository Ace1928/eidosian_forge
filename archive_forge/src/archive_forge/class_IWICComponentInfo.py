from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICComponentInfo(com.pIUnknown):
    _methods_ = [('GetComponentType', com.STDMETHOD()), ('GetCLSID', com.STDMETHOD()), ('GetSigningStatus', com.STDMETHOD()), ('GetAuthor', com.STDMETHOD()), ('GetVendorGUID', com.STDMETHOD()), ('GetVersion', com.STDMETHOD()), ('GetSpecVersion', com.STDMETHOD()), ('GetFriendlyName', com.STDMETHOD())]