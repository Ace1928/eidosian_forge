import ctypes
from pyglet.libs.win32 import com
class DS3DLISTENER(ctypes.Structure):
    _fields_ = [('dwSize', DWORD), ('vPosition', D3DVECTOR), ('vVelocity', D3DVECTOR), ('vOrientFront', D3DVECTOR), ('vOrientTop', D3DVECTOR), ('flDistanceFactor', D3DVALUE), ('flRolloffFactor', D3DVALUE), ('flDopplerFactor', D3DVALUE)]