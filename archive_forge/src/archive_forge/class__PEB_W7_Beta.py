from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class _PEB_W7_Beta(Structure):
    """
    This definition of the PEB structure is only valid for the beta versions
    of Windows 7. For the final version of Windows 7 use L{_PEB_W7} instead.
    This structure is not chosen automatically.
    """
    _pack_ = 8
    _fields_ = [('InheritedAddressSpace', BOOLEAN), ('ReadImageFileExecOptions', UCHAR), ('BeingDebugged', BOOLEAN), ('BitField', UCHAR), ('Mutant', HANDLE), ('ImageBaseAddress', PVOID), ('Ldr', PVOID), ('ProcessParameters', PVOID), ('SubSystemData', PVOID), ('ProcessHeap', PVOID), ('FastPebLock', PVOID), ('AtlThunkSListPtr', PVOID), ('IFEOKey', PVOID), ('CrossProcessFlags', DWORD), ('KernelCallbackTable', PVOID), ('SystemReserved', DWORD), ('TracingFlags', DWORD), ('ApiSetMap', PVOID), ('TlsExpansionCounter', DWORD), ('TlsBitmap', PVOID), ('TlsBitmapBits', DWORD * 2), ('ReadOnlySharedMemoryBase', PVOID), ('HotpatchInformation', PVOID), ('ReadOnlyStaticServerData', PVOID), ('AnsiCodePageData', PVOID), ('OemCodePageData', PVOID), ('UnicodeCaseTableData', PVOID), ('NumberOfProcessors', DWORD), ('NtGlobalFlag', DWORD), ('CriticalSectionTimeout', LONGLONG), ('HeapSegmentReserve', DWORD), ('HeapSegmentCommit', DWORD), ('HeapDeCommitTotalFreeThreshold', DWORD), ('HeapDeCommitFreeBlockThreshold', DWORD), ('NumberOfHeaps', DWORD), ('MaximumNumberOfHeaps', DWORD), ('ProcessHeaps', PVOID), ('GdiSharedHandleTable', PVOID), ('ProcessStarterHelper', PVOID), ('GdiDCAttributeList', DWORD), ('LoaderLock', PVOID), ('OSMajorVersion', DWORD), ('OSMinorVersion', DWORD), ('OSBuildNumber', WORD), ('OSCSDVersion', WORD), ('OSPlatformId', DWORD), ('ImageSubsystem', DWORD), ('ImageSubsystemMajorVersion', DWORD), ('ImageSubsystemMinorVersion', DWORD), ('ActiveProcessAffinityMask', DWORD), ('GdiHandleBuffer', DWORD * 34), ('PostProcessInitRoutine', PPS_POST_PROCESS_INIT_ROUTINE), ('TlsExpansionBitmap', PVOID), ('TlsExpansionBitmapBits', DWORD * 32), ('SessionId', DWORD), ('AppCompatFlags', ULONGLONG), ('AppCompatFlagsUser', ULONGLONG), ('pShimData', PVOID), ('AppCompatInfo', PVOID), ('CSDVersion', UNICODE_STRING), ('ActivationContextData', PVOID), ('ProcessAssemblyStorageMap', PVOID), ('SystemDefaultActivationContextData', PVOID), ('SystemAssemblyStorageMap', PVOID), ('MinimumStackCommit', DWORD), ('FlsCallback', PVOID), ('FlsListHead', LIST_ENTRY), ('FlsBitmap', PVOID), ('FlsBitmapBits', DWORD * 4), ('FlsHighIndex', DWORD), ('WerRegistrationData', PVOID), ('WerShipAssertPtr', PVOID), ('pContextData', PVOID), ('pImageHeaderHash', PVOID)]

    def __get_UserSharedInfoPtr(self):
        return self.KernelCallbackTable

    def __set_UserSharedInfoPtr(self, value):
        self.KernelCallbackTable = value
    UserSharedInfoPtr = property(__get_UserSharedInfoPtr, __set_UserSharedInfoPtr)