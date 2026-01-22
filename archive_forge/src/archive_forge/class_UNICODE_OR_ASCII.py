from ctypes import Union, Structure, c_char, c_short, c_long, c_ulong
from ctypes.wintypes import DWORD, BOOL, LPVOID, WORD, WCHAR
class UNICODE_OR_ASCII(Union):
    _fields_ = [('AsciiChar', c_char), ('UnicodeChar', WCHAR)]