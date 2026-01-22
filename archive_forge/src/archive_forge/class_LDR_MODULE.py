from winappdbg.win32.defines import *
from winappdbg.win32.version import os
class LDR_MODULE(Structure):
    _fields_ = [('InLoadOrderModuleList', LIST_ENTRY), ('InMemoryOrderModuleList', LIST_ENTRY), ('InInitializationOrderModuleList', LIST_ENTRY), ('BaseAddress', PVOID), ('EntryPoint', PVOID), ('SizeOfImage', ULONG), ('FullDllName', UNICODE_STRING), ('BaseDllName', UNICODE_STRING), ('Flags', ULONG), ('LoadCount', SHORT), ('TlsIndex', SHORT), ('HashTableEntry', LIST_ENTRY), ('TimeDateStamp', ULONG)]