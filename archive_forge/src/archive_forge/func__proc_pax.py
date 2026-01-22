from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def _proc_pax(self, tarfile):
    """Process an extended or global header as described in
           POSIX.1-2008.
        """
    buf = tarfile.fileobj.read(self._block(self.size))
    if self.type == XGLTYPE:
        pax_headers = tarfile.pax_headers
    else:
        pax_headers = tarfile.pax_headers.copy()
    match = re.search(b'\\d+ hdrcharset=([^\\n]+)\\n', buf)
    if match is not None:
        pax_headers['hdrcharset'] = match.group(1).decode('utf-8')
    hdrcharset = pax_headers.get('hdrcharset')
    if hdrcharset == 'BINARY':
        encoding = tarfile.encoding
    else:
        encoding = 'utf-8'
    regex = re.compile(b'(\\d+) ([^=]+)=')
    pos = 0
    while True:
        match = regex.match(buf, pos)
        if not match:
            break
        length, keyword = match.groups()
        length = int(length)
        if length == 0:
            raise InvalidHeaderError('invalid header')
        value = buf[match.end(2) + 1:match.start(1) + length - 1]
        keyword = self._decode_pax_field(keyword, 'utf-8', 'utf-8', tarfile.errors)
        if keyword in PAX_NAME_FIELDS:
            value = self._decode_pax_field(value, encoding, tarfile.encoding, tarfile.errors)
        else:
            value = self._decode_pax_field(value, 'utf-8', 'utf-8', tarfile.errors)
        pax_headers[keyword] = value
        pos += length
    try:
        next = self.fromtarfile(tarfile)
    except HeaderError as e:
        raise SubsequentHeaderError(str(e)) from None
    if 'GNU.sparse.map' in pax_headers:
        self._proc_gnusparse_01(next, pax_headers)
    elif 'GNU.sparse.size' in pax_headers:
        self._proc_gnusparse_00(next, pax_headers, buf)
    elif pax_headers.get('GNU.sparse.major') == '1' and pax_headers.get('GNU.sparse.minor') == '0':
        self._proc_gnusparse_10(next, pax_headers, tarfile)
    if self.type in (XHDTYPE, SOLARIS_XHDTYPE):
        next._apply_pax_info(pax_headers, tarfile.encoding, tarfile.errors)
        next.offset = self.offset
        if 'size' in pax_headers:
            offset = next.offset_data
            if next.isreg() or next.type not in SUPPORTED_TYPES:
                offset += next._block(next.size)
            tarfile.offset = offset
    return next