from winappdbg.win32.defines import *
from winappdbg.win32.version import *
from winappdbg.win32.kernel32 import *
def ImagehlpApiVersion():
    _ImagehlpApiVersion = windll.dbghelp.ImagehlpApiVersion
    _ImagehlpApiVersion.restype = LPAPI_VERSION
    api_version = _ImagehlpApiVersion()
    return api_version.contents