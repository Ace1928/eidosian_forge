from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
def NtQueryInformationFile(FileHandle, FileInformationClass, FileInformation, Length):
    _NtQueryInformationFile = windll.ntdll.NtQueryInformationFile
    _NtQueryInformationFile.argtypes = [HANDLE, PIO_STATUS_BLOCK, PVOID, ULONG, DWORD]
    _NtQueryInformationFile.restype = NTSTATUS
    IoStatusBlock = IO_STATUS_BLOCK()
    ntstatus = _NtQueryInformationFile(FileHandle, byref(IoStatusBlock), byref(FileInformation), Length, FileInformationClass)
    if ntstatus != 0:
        raise ctypes.WinError(RtlNtStatusToDosError(ntstatus))
    return IoStatusBlock