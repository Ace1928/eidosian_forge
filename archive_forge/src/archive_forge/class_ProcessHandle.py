import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
class ProcessHandle(Handle):
    """
    Win32 process handle.

    @type dwAccess: int
    @ivar dwAccess: Current access flags to this handle.
            This is the same value passed to L{OpenProcess}.
            Can only be C{None} if C{aHandle} is also C{None}.
            Defaults to L{PROCESS_ALL_ACCESS}.

    @see: L{Handle}
    """

    def __init__(self, aHandle=None, bOwnership=True, dwAccess=PROCESS_ALL_ACCESS):
        """
        @type  aHandle: int
        @param aHandle: Win32 handle value.

        @type  bOwnership: bool
        @param bOwnership:
           C{True} if we own the handle and we need to close it.
           C{False} if someone else will be calling L{CloseHandle}.

        @type  dwAccess: int
        @param dwAccess: Current access flags to this handle.
            This is the same value passed to L{OpenProcess}.
            Can only be C{None} if C{aHandle} is also C{None}.
            Defaults to L{PROCESS_ALL_ACCESS}.
        """
        super(ProcessHandle, self).__init__(aHandle, bOwnership)
        self.dwAccess = dwAccess
        if aHandle is not None and dwAccess is None:
            msg = 'Missing access flags for process handle: %x' % aHandle
            raise TypeError(msg)

    def get_pid(self):
        """
        @rtype:  int
        @return: Process global ID.
        """
        return GetProcessId(self.value)