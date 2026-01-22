from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class SaferLevelHandle(UserModeHandle):
    """
    Safer level handle.

    @see: U{http://msdn.microsoft.com/en-us/library/ms722425(VS.85).aspx}
    """
    _TYPE = SAFER_LEVEL_HANDLE

    def _close(self):
        SaferCloseLevel(self.value)