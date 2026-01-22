import ctypes
from pyglet.libs.win32 import com
class DSBCAPS(ctypes.Structure):
    _fields_ = [('dwSize', DWORD), ('dwFlags', DWORD), ('dwBufferBytes', DWORD), ('dwUnlockTransferRate', DWORD), ('dwPlayCpuOverhead', DWORD)]