from pyglet.image import *
from pyglet.image.codecs import *
from pyglet.libs.win32 import _kernel32 as kernel32
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
class IWICMetadataQueryReader(com.pIUnknown):
    _methods_ = [('GetContainerFormat', com.STDMETHOD()), ('GetLocation', com.STDMETHOD()), ('GetMetadataByName', com.STDMETHOD(LPCWSTR, c_void_p)), ('GetEnumerator', com.STDMETHOD())]