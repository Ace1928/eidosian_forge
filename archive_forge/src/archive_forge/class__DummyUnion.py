from ctypes import c_void_p
from ctypes import Structure
from ctypes import Union
from ctypes.wintypes import DWORD
from ctypes.wintypes import HANDLE
class _DummyUnion(Union):
    _fields_ = [('_offsets', _DummyStruct), ('Pointer', c_void_p)]