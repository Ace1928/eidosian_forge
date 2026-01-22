import builtins
import codecs
import datetime
import select
import struct
import sys
import zlib
import subunit
import iso8601
def _parse_varint(self, data, pos, max_3_bytes=False):
    data_0 = struct.unpack(FMT_8, data[pos:pos + 1])[0]
    typeenum = data_0 & 192
    value_0 = data_0 & 63
    if typeenum == 0:
        return (value_0, 1)
    elif typeenum == 64:
        data_1 = struct.unpack(FMT_8, data[pos + 1:pos + 2])[0]
        return (value_0 << 8 | data_1, 2)
    elif typeenum == 128:
        data_1 = struct.unpack(FMT_16, data[pos + 1:pos + 3])[0]
        return (value_0 << 16 | data_1, 3)
    else:
        if max_3_bytes:
            raise ParseError('3 byte maximum given but 4 byte value found.')
        data_1, data_2 = struct.unpack(FMT_24, data[pos + 1:pos + 4])
        result = value_0 << 24 | data_1 << 8 | data_2
        return (result, 4)