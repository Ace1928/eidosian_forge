from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
def NtQueryInformationProcess(ProcessHandle, ProcessInformationClass, ProcessInformationLength=None):
    _NtQueryInformationProcess = windll.ntdll.NtQueryInformationProcess
    _NtQueryInformationProcess.argtypes = [HANDLE, PROCESSINFOCLASS, PVOID, ULONG, PULONG]
    _NtQueryInformationProcess.restype = NTSTATUS
    if ProcessInformationLength is not None:
        ProcessInformation = ctypes.create_string_buffer('', ProcessInformationLength)
    elif ProcessInformationClass == ProcessBasicInformation:
        ProcessInformation = PROCESS_BASIC_INFORMATION()
        ProcessInformationLength = sizeof(PROCESS_BASIC_INFORMATION)
    elif ProcessInformationClass == ProcessImageFileName:
        unicode_buffer = ctypes.create_unicode_buffer(u'', 4096)
        ProcessInformation = UNICODE_STRING(0, 4096, addressof(unicode_buffer))
        ProcessInformationLength = sizeof(UNICODE_STRING)
    elif ProcessInformationClass in (ProcessDebugPort, ProcessWow64Information, ProcessWx86Information, ProcessHandleCount, ProcessPriorityBoost):
        ProcessInformation = DWORD()
        ProcessInformationLength = sizeof(DWORD)
    else:
        raise Exception('Unknown ProcessInformationClass, use an explicit ProcessInformationLength value instead')
    ReturnLength = ULONG(0)
    ntstatus = _NtQueryInformationProcess(ProcessHandle, ProcessInformationClass, byref(ProcessInformation), ProcessInformationLength, byref(ReturnLength))
    if ntstatus != 0:
        raise ctypes.WinError(RtlNtStatusToDosError(ntstatus))
    if ProcessInformationClass == ProcessBasicInformation:
        retval = ProcessInformation
    elif ProcessInformationClass in (ProcessDebugPort, ProcessWow64Information, ProcessWx86Information, ProcessHandleCount, ProcessPriorityBoost):
        retval = ProcessInformation.value
    elif ProcessInformationClass == ProcessImageFileName:
        vptr = ctypes.c_void_p(ProcessInformation.Buffer)
        cptr = ctypes.cast(vptr, ctypes.c_wchar * ProcessInformation.Length)
        retval = cptr.contents.raw
    else:
        retval = ProcessInformation.raw[:ReturnLength.value]
    return retval