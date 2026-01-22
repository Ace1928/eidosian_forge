import ctypes
from pyglet.libs.win32 import com
class IDirectSound(com.pIUnknown):
    _methods_ = [('CreateSoundBuffer', com.STDMETHOD(LPDSBUFFERDESC, ctypes.POINTER(IDirectSoundBuffer), LPUNKNOWN)), ('GetCaps', com.STDMETHOD(LPDSCAPS)), ('DuplicateSoundBuffer', com.STDMETHOD(IDirectSoundBuffer, ctypes.POINTER(IDirectSoundBuffer))), ('SetCooperativeLevel', com.STDMETHOD(HWND, DWORD)), ('Compact', com.STDMETHOD()), ('GetSpeakerConfig', com.STDMETHOD(LPDWORD)), ('SetSpeakerConfig', com.STDMETHOD(DWORD)), ('Initialize', com.STDMETHOD(com.LPGUID))]