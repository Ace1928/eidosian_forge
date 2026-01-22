import calendar
from datetime import datetime
from decimal import Decimal
from io import BytesIO
from struct import pack, unpack_from
from .exceptions import FrameSyntaxError
from .spec import Basic
from .utils import bytes_to_str as pstr_t
from .utils import str_to_bytes
def _read_item(buf, offset):
    ftype = chr(buf[offset])
    offset += 1
    if ftype == 'S':
        slen, = unpack_from('>I', buf, offset)
        offset += 4
        try:
            val = pstr_t(buf[offset:offset + slen])
        except UnicodeDecodeError:
            val = buf[offset:offset + slen]
        offset += slen
    elif ftype == 's':
        slen, = unpack_from('>B', buf, offset)
        offset += 1
        val = pstr_t(buf[offset:offset + slen])
        offset += slen
    elif ftype == 'x':
        blen, = unpack_from('>I', buf, offset)
        offset += 4
        val = buf[offset:offset + blen]
        offset += blen
    elif ftype == 'b':
        val, = unpack_from('>B', buf, offset)
        offset += 1
    elif ftype == 'B':
        val, = unpack_from('>b', buf, offset)
        offset += 1
    elif ftype == 'U':
        val, = unpack_from('>h', buf, offset)
        offset += 2
    elif ftype == 'u':
        val, = unpack_from('>H', buf, offset)
        offset += 2
    elif ftype == 'I':
        val, = unpack_from('>i', buf, offset)
        offset += 4
    elif ftype == 'i':
        val, = unpack_from('>I', buf, offset)
        offset += 4
    elif ftype == 'L':
        val, = unpack_from('>q', buf, offset)
        offset += 8
    elif ftype == 'l':
        val, = unpack_from('>Q', buf, offset)
        offset += 8
    elif ftype == 'f':
        val, = unpack_from('>f', buf, offset)
        offset += 4
    elif ftype == 'd':
        val, = unpack_from('>d', buf, offset)
        offset += 8
    elif ftype == 'D':
        d, = unpack_from('>B', buf, offset)
        offset += 1
        n, = unpack_from('>i', buf, offset)
        offset += 4
        val = Decimal(n) / Decimal(10 ** d)
    elif ftype == 'F':
        tlen, = unpack_from('>I', buf, offset)
        offset += 4
        limit = offset + tlen
        val = {}
        while offset < limit:
            keylen, = unpack_from('>B', buf, offset)
            offset += 1
            key = pstr_t(buf[offset:offset + keylen])
            offset += keylen
            val[key], offset = _read_item(buf, offset)
    elif ftype == 'A':
        alen, = unpack_from('>I', buf, offset)
        offset += 4
        limit = offset + alen
        val = []
        while offset < limit:
            v, offset = _read_item(buf, offset)
            val.append(v)
    elif ftype == 't':
        val, = unpack_from('>B', buf, offset)
        val = bool(val)
        offset += 1
    elif ftype == 'T':
        val, = unpack_from('>Q', buf, offset)
        offset += 8
        val = datetime.utcfromtimestamp(val)
    elif ftype == 'V':
        val = None
    else:
        raise FrameSyntaxError('Unknown value in table: {!r} ({!r})'.format(ftype, type(ftype)))
    return (val, offset)