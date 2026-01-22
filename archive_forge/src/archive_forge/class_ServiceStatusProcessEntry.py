from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
class ServiceStatusProcessEntry(object):
    """
    Service status entry returned by L{EnumServicesStatusEx}.
    """

    def __init__(self, raw):
        """
        @type  raw: L{ENUM_SERVICE_STATUS_PROCESSA} or L{ENUM_SERVICE_STATUS_PROCESSW}
        @param raw: Raw structure for this service status entry.
        """
        self.ServiceName = raw.lpServiceName
        self.DisplayName = raw.lpDisplayName
        self.ServiceType = raw.ServiceStatusProcess.dwServiceType
        self.CurrentState = raw.ServiceStatusProcess.dwCurrentState
        self.ControlsAccepted = raw.ServiceStatusProcess.dwControlsAccepted
        self.Win32ExitCode = raw.ServiceStatusProcess.dwWin32ExitCode
        self.ServiceSpecificExitCode = raw.ServiceStatusProcess.dwServiceSpecificExitCode
        self.CheckPoint = raw.ServiceStatusProcess.dwCheckPoint
        self.WaitHint = raw.ServiceStatusProcess.dwWaitHint
        self.ProcessId = raw.ServiceStatusProcess.dwProcessId
        self.ServiceFlags = raw.ServiceStatusProcess.dwServiceFlags

    def __str__(self):
        output = []
        if self.ServiceType & SERVICE_INTERACTIVE_PROCESS:
            output.append('Interactive service ')
        else:
            output.append('Service ')
        if self.DisplayName:
            output.append('"%s" (%s)' % (self.DisplayName, self.ServiceName))
        else:
            output.append('"%s"' % self.ServiceName)
        if self.CurrentState == SERVICE_CONTINUE_PENDING:
            output.append(' is about to continue')
        elif self.CurrentState == SERVICE_PAUSE_PENDING:
            output.append(' is pausing')
        elif self.CurrentState == SERVICE_PAUSED:
            output.append(' is paused')
        elif self.CurrentState == SERVICE_RUNNING:
            output.append(' is running')
        elif self.CurrentState == SERVICE_START_PENDING:
            output.append(' is starting')
        elif self.CurrentState == SERVICE_STOP_PENDING:
            output.append(' is stopping')
        elif self.CurrentState == SERVICE_STOPPED:
            output.append(' is stopped')
        if self.ProcessId:
            output.append(' at process %d' % self.ProcessId)
        output.append('.')
        return ''.join(output)