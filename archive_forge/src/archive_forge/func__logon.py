import os
from ctypes import *
def _logon(profileName=None, password=None):
    pSession = LHANDLE()
    rc = MAPILogon(0, profileName, password, MAPI_LOGON_UI, 0, byref(pSession))
    if rc != SUCCESS_SUCCESS:
        raise MAPIError(rc)
    return pSession