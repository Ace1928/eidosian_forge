from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def ImagehlpApiVersionEx(MajorVersion, MinorVersion, Revision):
    _ImagehlpApiVersionEx = windll.dbghelp.ImagehlpApiVersionEx
    _ImagehlpApiVersionEx.argtypes = [LPAPI_VERSION]
    _ImagehlpApiVersionEx.restype = LPAPI_VERSION
    api_version = API_VERSION(MajorVersion, MinorVersion, Revision, 0)
    ret_api_version = _ImagehlpApiVersionEx(byref(api_version))
    return ret_api_version.contents