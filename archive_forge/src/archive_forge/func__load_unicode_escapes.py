import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
def _load_unicode_escapes(v, hexbytes, prefix):
    skip = False
    i = len(v) - 1
    while i > -1 and v[i] == '\\':
        skip = not skip
        i -= 1
    for hx in hexbytes:
        if skip:
            skip = False
            i = len(hx) - 1
            while i > -1 and hx[i] == '\\':
                skip = not skip
                i -= 1
            v += prefix
            v += hx
            continue
        hxb = ''
        i = 0
        hxblen = 4
        if prefix == '\\U':
            hxblen = 8
        hxb = ''.join(hx[i:i + hxblen]).lower()
        if hxb.strip('0123456789abcdef'):
            raise ValueError('Invalid escape sequence: ' + hxb)
        if hxb[0] == 'd' and hxb[1].strip('01234567'):
            raise ValueError('Invalid escape sequence: ' + hxb + '. Only scalar unicode points are allowed.')
        v += unichr(int(hxb, 16))
        v += unicode(hx[len(hxb):])
    return v