from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
class IMAGEHLP_MODULE(Structure):
    _fields_ = [('SizeOfStruct', DWORD), ('BaseOfImage', DWORD), ('ImageSize', DWORD), ('TimeDateStamp', DWORD), ('CheckSum', DWORD), ('NumSyms', DWORD), ('SymType', DWORD), ('ModuleName', CHAR * 32), ('ImageName', CHAR * 256), ('LoadedImageName', CHAR * 256)]