import sys
def _crc24r(data, crc, table):
    crc = crc & 16777215
    for x in data:
        crc = table[ord(x) ^ int(crc & 255)] ^ crc >> 8
    return crc