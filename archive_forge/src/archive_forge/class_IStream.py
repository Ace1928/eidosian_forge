import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class IStream(com.pIUnknown):
    _methods_ = [('Read', com.STDMETHOD(c_void_p, ULONG, POINTER(ULONG))), ('Write', com.STDMETHOD()), ('Seek', com.STDMETHOD(LARGE_INTEGER, DWORD, POINTER(ULARGE_INTEGER))), ('SetSize', com.STDMETHOD()), ('CopyTo', com.STDMETHOD()), ('Commit', com.STDMETHOD()), ('Revert', com.STDMETHOD()), ('LockRegion', com.STDMETHOD()), ('UnlockRegion', com.STDMETHOD()), ('Stat', com.STDMETHOD(POINTER(STATSTG), UINT)), ('Clone', com.STDMETHOD())]