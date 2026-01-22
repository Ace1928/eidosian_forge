from winappdbg.win32.defines import *
def EnumProcesses():
    _EnumProcesses = windll.psapi.EnumProcesses
    _EnumProcesses.argtypes = [LPVOID, DWORD, LPDWORD]
    _EnumProcesses.restype = bool
    _EnumProcesses.errcheck = RaiseIfZero
    size = 4096
    cbBytesReturned = DWORD()
    unit = sizeof(DWORD)
    while 1:
        ProcessIds = (DWORD * (size // unit))()
        cbBytesReturned.value = size
        _EnumProcesses(byref(ProcessIds), cbBytesReturned, byref(cbBytesReturned))
        returned = cbBytesReturned.value
        if returned < size:
            break
        size = size + 4096
    ProcessIdList = list()
    for ProcessId in ProcessIds:
        if ProcessId is None:
            break
        ProcessIdList.append(ProcessId)
    return ProcessIdList