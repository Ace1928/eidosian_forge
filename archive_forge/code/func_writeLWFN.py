import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re
def writeLWFN(path, data):
    Res.FSpCreateResFile(path, 'just', 'LWFN', 0)
    resRef = Res.FSOpenResFile(path, 2)
    try:
        Res.UseResFile(resRef)
        resID = 501
        chunks = findEncryptedChunks(data)
        for isEncrypted, chunk in chunks:
            if isEncrypted:
                code = 2
            else:
                code = 1
            while chunk:
                res = Res.Resource(bytechr(code) + '\x00' + chunk[:LWFNCHUNKSIZE - 2])
                res.AddResource('POST', resID, '')
                chunk = chunk[LWFNCHUNKSIZE - 2:]
                resID = resID + 1
        res = Res.Resource(bytechr(5) + '\x00')
        res.AddResource('POST', resID, '')
    finally:
        Res.CloseResFile(resRef)