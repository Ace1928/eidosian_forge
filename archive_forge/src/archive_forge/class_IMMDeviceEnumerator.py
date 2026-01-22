from typing import List, Optional, Tuple
from pyglet.libs.win32 import com
from pyglet.libs.win32 import _ole32 as ole32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.libs.win32.types import *
from pyglet.media.devices import base
from pyglet.util import debug_print
class IMMDeviceEnumerator(com.pIUnknown):
    _methods_ = [('EnumAudioEndpoints', com.STDMETHOD(EDataFlow, DWORD, c_void_p)), ('GetDefaultAudioEndpoint', com.STDMETHOD(EDataFlow, ERole, ctypes.POINTER(IMMDevice))), ('GetDevice', com.STDMETHOD(LPCWSTR, POINTER(IMMDevice))), ('RegisterEndpointNotificationCallback', com.STDMETHOD(POINTER(IMMNotificationClient))), ('UnregisterEndpointNotificationCallback', com.STDMETHOD())]