from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
def NtQueryInformationThread(ThreadHandle, ThreadInformationClass, ThreadInformationLength=None):
    _NtQueryInformationThread = windll.ntdll.NtQueryInformationThread
    _NtQueryInformationThread.argtypes = [HANDLE, THREADINFOCLASS, PVOID, ULONG, PULONG]
    _NtQueryInformationThread.restype = NTSTATUS
    if ThreadInformationLength is not None:
        ThreadInformation = ctypes.create_string_buffer('', ThreadInformationLength)
    else:
        if ThreadInformationClass == ThreadBasicInformation:
            ThreadInformation = THREAD_BASIC_INFORMATION()
        elif ThreadInformationClass == ThreadHideFromDebugger:
            ThreadInformation = BOOLEAN()
        elif ThreadInformationClass == ThreadQuerySetWin32StartAddress:
            ThreadInformation = PVOID()
        elif ThreadInformationClass in (ThreadAmILastThread, ThreadPriorityBoost):
            ThreadInformation = DWORD()
        elif ThreadInformationClass == ThreadPerformanceCount:
            ThreadInformation = LONGLONG()
        else:
            raise Exception('Unknown ThreadInformationClass, use an explicit ThreadInformationLength value instead')
        ThreadInformationLength = sizeof(ThreadInformation)
    ReturnLength = ULONG(0)
    ntstatus = _NtQueryInformationThread(ThreadHandle, ThreadInformationClass, byref(ThreadInformation), ThreadInformationLength, byref(ReturnLength))
    if ntstatus != 0:
        raise ctypes.WinError(RtlNtStatusToDosError(ntstatus))
    if ThreadInformationClass == ThreadBasicInformation:
        retval = ThreadInformation
    elif ThreadInformationClass == ThreadHideFromDebugger:
        retval = bool(ThreadInformation.value)
    elif ThreadInformationClass in (ThreadQuerySetWin32StartAddress, ThreadAmILastThread, ThreadPriorityBoost, ThreadPerformanceCount):
        retval = ThreadInformation.value
    else:
        retval = ThreadInformation.raw[:ReturnLength.value]
    return retval