import os
from ctypes import *
def _sendMail(session, recipient, subject, body, attach):
    nFileCount = len(attach)
    if attach:
        MapiFileDesc_A = MapiFileDesc * len(attach)
        fda = MapiFileDesc_A()
        for fd, fa in zip(fda, attach):
            fd.ulReserved = 0
            fd.flFlags = 0
            fd.nPosition = -1
            fd.lpszPathName = fa
            fd.lpszFileName = None
            fd.lpFileType = None
        lpFiles = fda
    else:
        lpFiles = lpMapiFileDesc()
    RecipWork = recipient.split(';')
    RecipCnt = len(RecipWork)
    MapiRecipDesc_A = MapiRecipDesc * len(RecipWork)
    rda = MapiRecipDesc_A()
    for rd, ra in zip(rda, RecipWork):
        rd.ulReserved = 0
        rd.ulRecipClass = MAPI_TO
        try:
            rd.lpszName, rd.lpszAddress = _resolveName(session, ra)
        except OSError:
            rd.lpszName, rd.lpszAddress = (None, ra)
        rd.ulEIDSize = 0
        rd.lpEntryID = None
    recip = rda
    msg = MapiMessage(0, subject, body, None, None, None, 0, lpMapiRecipDesc(), RecipCnt, recip, nFileCount, lpFiles)
    rc = MAPISendMail(session, 0, byref(msg), MAPI_DIALOG, 0)
    if rc != SUCCESS_SUCCESS:
        raise MAPIError(rc)