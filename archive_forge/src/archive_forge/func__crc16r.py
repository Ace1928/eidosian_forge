import sys
def _crc16r(data, crc, table):
    crc = crc & 65535
    for x in data:
        crc = table[ord(x) ^ crc & 255] ^ crc >> 8
    return crc