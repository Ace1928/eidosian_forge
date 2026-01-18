import fontTools
from fontTools.misc import eexec
from fontTools.misc.macCreatorType import getMacCreatorAndType
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes
from fontTools.misc.psOperators import (
from fontTools.encodings.StandardEncoding import StandardEncoding
import os
import re
def writePFB(path, data):
    chunks = findEncryptedChunks(data)
    with open(path, 'wb') as f:
        for isEncrypted, chunk in chunks:
            if isEncrypted:
                code = 2
            else:
                code = 1
            f.write(bytechr(128) + bytechr(code))
            f.write(longToString(len(chunk)))
            f.write(chunk)
        f.write(bytechr(128) + bytechr(3))