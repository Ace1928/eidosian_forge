import builtins
import codecs
import datetime
import select
import struct
import sys
import zlib
import subunit
import iso8601
def _read_utf8(self, buf, pos):
    length, consumed = self._parse_varint(buf, pos)
    pos += consumed
    utf8_bytes = buf[pos:pos + length]
    if length != len(utf8_bytes):
        raise ParseError('UTF8 string at offset %d extends past end of packet: claimed %d bytes, %d available' % (pos - 2, length, len(utf8_bytes)))
    if NUL_ELEMENT in utf8_bytes:
        raise ParseError('UTF8 string at offset %d contains NUL byte' % (pos - 2,))
    try:
        utf8, decoded_bytes = utf_8_decode(utf8_bytes)
        if decoded_bytes != length:
            raise ParseError('Invalid (partially decodable) string at offset %d, %d undecoded bytes' % (pos - 2, length - decoded_bytes))
        return (utf8, length + pos)
    except UnicodeDecodeError:
        raise ParseError('UTF8 string at offset %d is not UTF8' % (pos - 2,))