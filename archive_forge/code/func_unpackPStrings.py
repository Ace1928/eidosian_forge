from fontTools import ttLib
from fontTools.ttLib.standardGlyphOrder import standardGlyphOrder
from fontTools.misc import sstruct
from fontTools.misc.textTools import bytechr, byteord, tobytes, tostr, safeEval, readHex
from . import DefaultTable
import sys
import struct
import array
import logging
def unpackPStrings(data, n):
    strings = []
    index = 0
    dataLen = len(data)
    for _ in range(n):
        if dataLen <= index:
            length = 0
        else:
            length = byteord(data[index])
        index += 1
        if dataLen <= index + length - 1:
            name = ''
        else:
            name = tostr(data[index:index + length], encoding='latin1')
        strings.append(name)
        index += length
    if index < dataLen:
        log.warning('%d extra bytes in post.stringData array', dataLen - index)
    elif dataLen < index:
        log.warning('not enough data in post.stringData array')
    return strings