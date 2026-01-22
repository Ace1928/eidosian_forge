import time
import weakref
import threading
import pyglet
from pyglet.libs.win32 import com
from pyglet.event import EventDispatcher
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import _ole32 as ole32, _oleaut32 as oleaut32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.input.base import Device, Controller, Button, AbsoluteAxis, ControllerManager
class IWbemServices(com.pIUnknown):
    _methods_ = [('OpenNamespace', com.STDMETHOD()), ('CancelAsyncCall', com.STDMETHOD()), ('QueryObjectSink', com.STDMETHOD()), ('GetObject', com.STDMETHOD()), ('GetObjectAsync', com.STDMETHOD()), ('PutClass', com.STDMETHOD()), ('PutClassAsync', com.STDMETHOD()), ('DeleteClass', com.STDMETHOD()), ('DeleteClassAsync', com.STDMETHOD()), ('CreateClassEnum', com.STDMETHOD()), ('CreateClassEnumAsync', com.STDMETHOD()), ('PutInstance', com.STDMETHOD()), ('PutInstanceAsync', com.STDMETHOD()), ('DeleteInstance', com.STDMETHOD()), ('DeleteInstanceAsync', com.STDMETHOD()), ('CreateInstanceEnum', com.STDMETHOD(BSTR, LONG, IWbemContext, POINTER(IEnumWbemClassObject))), ('CreateInstanceEnumAsync', com.STDMETHOD())]