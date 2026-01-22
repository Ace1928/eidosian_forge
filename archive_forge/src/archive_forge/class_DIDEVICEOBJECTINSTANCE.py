import ctypes
from pyglet.libs.win32 import com
class DIDEVICEOBJECTINSTANCE(ctypes.Structure):
    _fields_ = (('dwSize', DWORD), ('guidType', com.GUID), ('dwOfs', DWORD), ('dwType', DWORD), ('dwFlags', DWORD), ('tszName', WCHAR * MAX_PATH), ('dwFFMaxForce', DWORD), ('dwFFForceResolution', DWORD), ('wCollectionNumber', WORD), ('wDesignatorIndex', WORD), ('wUsagePage', WORD), ('wUsage', WORD), ('dwDimension', DWORD), ('wExponent', WORD), ('wReportId', WORD))