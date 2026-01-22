from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
class PdfDict(collections.UserDict):

    def __setattr__(self, key, value):
        if key == 'data':
            collections.UserDict.__setattr__(self, key, value)
        else:
            self[key.encode('us-ascii')] = value

    def __getattr__(self, key):
        try:
            value = self[key.encode('us-ascii')]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(value, bytes):
            value = decode_text(value)
        if key.endswith('Date'):
            if value.startswith('D:'):
                value = value[2:]
            relationship = 'Z'
            if len(value) > 17:
                relationship = value[14]
                offset = int(value[15:17]) * 60
                if len(value) > 20:
                    offset += int(value[18:20])
            format = '%Y%m%d%H%M%S'[:len(value) - 2]
            value = time.strptime(value[:len(format) + 2], format)
            if relationship in ['+', '-']:
                offset *= 60
                if relationship == '+':
                    offset *= -1
                value = time.gmtime(calendar.timegm(value) + offset)
        return value

    def __bytes__(self):
        out = bytearray(b'<<')
        for key, value in self.items():
            if value is None:
                continue
            value = pdf_repr(value)
            out.extend(b'\n')
            out.extend(bytes(PdfName(key)))
            out.extend(b' ')
            out.extend(value)
        out.extend(b'\n>>')
        return bytes(out)