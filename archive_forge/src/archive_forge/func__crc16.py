import sys
def _crc16(data, crc, table):
    crc = crc & 65535
    for x in data:
        crc = table[ord(x) ^ crc >> 8 & 255] ^ crc << 8 & 65280
    return crc