from ctypes import Union, Structure, c_char, c_short, c_long, c_ulong
from ctypes.wintypes import DWORD, BOOL, LPVOID, WORD, WCHAR
class SMALL_RECT(Structure):
    """struct in wincon.h."""
    _fields_ = [('Left', c_short), ('Top', c_short), ('Right', c_short), ('Bottom', c_short)]