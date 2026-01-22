from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def _internal_GetTokenInformation(hTokenHandle, TokenInformationClass, TokenInformation):
    _GetTokenInformation = windll.advapi32.GetTokenInformation
    _GetTokenInformation.argtypes = [HANDLE, TOKEN_INFORMATION_CLASS, LPVOID, DWORD, PDWORD]
    _GetTokenInformation.restype = bool
    _GetTokenInformation.errcheck = RaiseIfZero
    ReturnLength = DWORD(0)
    TokenInformationLength = SIZEOF(TokenInformation)
    _GetTokenInformation(hTokenHandle, TokenInformationClass, byref(TokenInformation), TokenInformationLength, byref(ReturnLength))
    if ReturnLength.value != TokenInformationLength:
        raise ctypes.WinError(ERROR_INSUFFICIENT_BUFFER)
    return TokenInformation