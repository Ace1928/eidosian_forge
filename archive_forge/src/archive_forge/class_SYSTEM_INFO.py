from winappdbg.win32.defines import *
class SYSTEM_INFO(Structure):
    _fields_ = [('id', _SYSTEM_INFO_OEM_ID), ('dwPageSize', DWORD), ('lpMinimumApplicationAddress', LPVOID), ('lpMaximumApplicationAddress', LPVOID), ('dwActiveProcessorMask', DWORD_PTR), ('dwNumberOfProcessors', DWORD), ('dwProcessorType', DWORD), ('dwAllocationGranularity', DWORD), ('wProcessorLevel', WORD), ('wProcessorRevision', WORD)]

    def __get_dwOemId(self):
        return self.id.dwOemId

    def __set_dwOemId(self, value):
        self.id.dwOemId = value
    dwOemId = property(__get_dwOemId, __set_dwOemId)

    def __get_wProcessorArchitecture(self):
        return self.id.w.wProcessorArchitecture

    def __set_wProcessorArchitecture(self, value):
        self.id.w.wProcessorArchitecture = value
    wProcessorArchitecture = property(__get_wProcessorArchitecture, __set_wProcessorArchitecture)