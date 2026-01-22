import os
from ctypes import *
def _resolveName(session, name):
    pRecipDesc = lpMapiRecipDesc()
    rc = MAPIResolveName(session, 0, name, 0, 0, byref(pRecipDesc))
    if rc != SUCCESS_SUCCESS:
        raise MAPIError(rc)
    rd = pRecipDesc.contents
    name, address = (rd.lpszName, rd.lpszAddress)
    rc = MAPIFreeBuffer(pRecipDesc)
    if rc != SUCCESS_SUCCESS:
        raise MAPIError(rc)
    return (name, address)