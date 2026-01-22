import ctypes
import ctypes.wintypes
import stat as stdstat
from collections import namedtuple
class BY_HANDLE_FILE_INFORMATION(ctypes.Structure):
    _fields_ = [('dwFileAttributes', ctypes.wintypes.DWORD), ('ftCreationTime', FILETIME), ('ftLastAccessTime', FILETIME), ('ftLastWriteTime', FILETIME), ('dwVolumeSerialNumber', ctypes.wintypes.DWORD), ('nFileSizeHigh', ctypes.wintypes.DWORD), ('nFileSizeLow', ctypes.wintypes.DWORD), ('nNumberOfLinks', ctypes.wintypes.DWORD), ('nFileIndexHigh', ctypes.wintypes.DWORD), ('nFileIndexLow', ctypes.wintypes.DWORD)]