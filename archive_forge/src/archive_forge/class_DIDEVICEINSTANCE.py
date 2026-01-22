import ctypes
from pyglet.libs.win32 import com
class DIDEVICEINSTANCE(ctypes.Structure):
    _fields_ = (('dwSize', DWORD), ('guidInstance', com.GUID), ('guidProduct', com.GUID), ('dwDevType', DWORD), ('tszInstanceName', WCHAR * MAX_PATH), ('tszProductName', WCHAR * MAX_PATH), ('guidFFDriver', com.GUID), ('wUsagePage', WORD), ('wUsage', WORD))