from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class ServiceControlManagerHandle(UserModeHandle):
    """
    Service Control Manager (SCM) handle.

    @see: U{http://msdn.microsoft.com/en-us/library/windows/desktop/ms684323(v=vs.85).aspx}
    """
    _TYPE = SC_HANDLE

    def _close(self):
        CloseServiceHandle(self.value)