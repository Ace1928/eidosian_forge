import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class _DEBUG_EVENT_UNION_(Union):
    _fields_ = [('Exception', EXCEPTION_DEBUG_INFO), ('CreateThread', CREATE_THREAD_DEBUG_INFO), ('CreateProcessInfo', CREATE_PROCESS_DEBUG_INFO), ('ExitThread', EXIT_THREAD_DEBUG_INFO), ('ExitProcess', EXIT_PROCESS_DEBUG_INFO), ('LoadDll', LOAD_DLL_DEBUG_INFO), ('UnloadDll', UNLOAD_DLL_DEBUG_INFO), ('DebugString', OUTPUT_DEBUG_STRING_INFO), ('RipInfo', RIP_INFO)]