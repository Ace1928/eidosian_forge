from ctypes import Union, Structure, c_char, c_short, c_long, c_ulong
from ctypes.wintypes import DWORD, BOOL, LPVOID, WORD, WCHAR
class INPUT_RECORD(Structure):
    """
    http://msdn.microsoft.com/en-us/library/windows/desktop/ms683499(v=vs.85).aspx
    """
    _fields_ = [('EventType', c_short), ('Event', EVENT_RECORD)]