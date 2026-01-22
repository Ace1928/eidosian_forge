from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class ServiceStatusProcess(object):
    """
    Wrapper for the L{SERVICE_STATUS_PROCESS} structure.
    """

    def __init__(self, raw):
        """
        @type  raw: L{SERVICE_STATUS_PROCESS}
        @param raw: Raw structure for this service status data.
        """
        self.ServiceType = raw.dwServiceType
        self.CurrentState = raw.dwCurrentState
        self.ControlsAccepted = raw.dwControlsAccepted
        self.Win32ExitCode = raw.dwWin32ExitCode
        self.ServiceSpecificExitCode = raw.dwServiceSpecificExitCode
        self.CheckPoint = raw.dwCheckPoint
        self.WaitHint = raw.dwWaitHint
        self.ProcessId = raw.dwProcessId
        self.ServiceFlags = raw.dwServiceFlags