import struct
import sys
def _mkCrcFun(poly, sizeBits, initCrc, rev, xorOut):
    if rev:
        tableList = _mkTable_r(poly, sizeBits)
        _fun = _sizeMap[sizeBits][1]
    else:
        tableList = _mkTable(poly, sizeBits)
        _fun = _sizeMap[sizeBits][0]
    _table = tableList
    if _usingExtension:
        _table = struct.pack(_sizeToTypeCode[sizeBits], *tableList)
    if xorOut == 0:

        def crcfun(data, crc=initCrc, table=_table, fun=_fun):
            return fun(data, crc, table)
    else:

        def crcfun(data, crc=initCrc, table=_table, fun=_fun):
            return xorOut ^ fun(data, xorOut ^ crc, table)
    return (crcfun, tableList)