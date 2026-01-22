from winappdbg.win32.defines import *
from winappdbg.win32.peb_teb import *
def NtSystemDebugControl(Command, InputBuffer=None, InputBufferLength=None, OutputBuffer=None, OutputBufferLength=None):
    _NtSystemDebugControl = windll.ntdll.NtSystemDebugControl
    _NtSystemDebugControl.argtypes = [SYSDBG_COMMAND, PVOID, ULONG, PVOID, ULONG, PULONG]
    _NtSystemDebugControl.restype = NTSTATUS
    if InputBuffer is None:
        if InputBufferLength is None:
            InputBufferLength = 0
        else:
            raise ValueError('Invalid call to NtSystemDebugControl: input buffer length given but no input buffer!')
    else:
        if InputBufferLength is None:
            InputBufferLength = sizeof(InputBuffer)
        InputBuffer = byref(InputBuffer)
    if OutputBuffer is None:
        if OutputBufferLength is None:
            OutputBufferLength = 0
        else:
            OutputBuffer = ctypes.create_string_buffer('', OutputBufferLength)
    elif OutputBufferLength is None:
        OutputBufferLength = sizeof(OutputBuffer)
    if OutputBuffer is not None:
        ReturnLength = ULONG(0)
        ntstatus = _NtSystemDebugControl(Command, InputBuffer, InputBufferLength, byref(OutputBuffer), OutputBufferLength, byref(ReturnLength))
        if ntstatus != 0:
            raise ctypes.WinError(RtlNtStatusToDosError(ntstatus))
        ReturnLength = ReturnLength.value
        if ReturnLength != OutputBufferLength:
            raise ctypes.WinError(ERROR_BAD_LENGTH)
        return (OutputBuffer, ReturnLength)
    ntstatus = _NtSystemDebugControl(Command, InputBuffer, InputBufferLength, OutputBuffer, OutputBufferLength, None)
    if ntstatus != 0:
        raise ctypes.WinError(RtlNtStatusToDosError(ntstatus))