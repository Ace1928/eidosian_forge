from __future__ import division, print_function
import sys
import os
import io
import re
import glob
import math
import zlib
import time
import json
import enum
import struct
import pathlib
import warnings
import binascii
import tempfile
import datetime
import threading
import collections
import multiprocessing
import concurrent.futures
import numpy
def _write_remaining_pages(self):
    """Write outstanding IFDs and tags to file."""
    if not self._tags or self._truncate:
        return
    fh = self._fh
    fhpos = fh.tell()
    if fhpos % 2:
        fh.write(b'\x00')
        fhpos += 1
    byteorder = self._byteorder
    offsetformat = self._offsetformat
    offsetsize = self._offsetsize
    tagnoformat = self._tagnoformat
    tagsize = self._tagsize
    dataoffset = self._dataoffset
    pagedatasize = sum(self._databytecounts)
    pageno = self._shape[0] * self._datashape[0] - 1

    def pack(fmt, *val):
        return struct.pack(byteorder + fmt, *val)
    ifd = io.BytesIO()
    ifd.write(pack(tagnoformat, len(self._tags)))
    tagoffset = ifd.tell()
    ifd.write(b''.join((t[1] for t in self._tags)))
    ifdoffset = ifd.tell()
    ifd.write(pack(offsetformat, 0))
    for tagindex, tag in enumerate(self._tags):
        offset2value = tagoffset + tagindex * tagsize + offsetsize + 4
        if tag[2]:
            pos = ifd.tell()
            if pos % 2:
                ifd.write(b'\x00')
                pos += 1
            ifd.seek(offset2value)
            try:
                ifd.write(pack(offsetformat, pos + fhpos))
            except Exception:
                if self._imagej:
                    warnings.warn('truncating ImageJ file')
                    self._truncate = True
                    return
                raise ValueError('data too large for non-BigTIFF file')
            ifd.seek(pos)
            ifd.write(tag[2])
            if tag[0] == self._tagoffsets:
                stripoffset2offset = offset2value
                stripoffset2value = pos
        elif tag[0] == self._tagoffsets:
            stripoffset2offset = None
            stripoffset2value = offset2value
    if ifd.tell() % 2:
        ifd.write(b'\x00')
    pos = fh.tell()
    if not self._bigtiff and pos + ifd.tell() * pageno > 2 ** 32 - 256:
        if self._imagej:
            warnings.warn('truncating ImageJ file')
            self._truncate = True
            return
        raise ValueError('data too large for non-BigTIFF file')
    for _ in range(pageno):
        pos = fh.tell()
        fh.seek(self._ifdoffset)
        fh.write(pack(offsetformat, pos))
        fh.seek(pos)
        self._ifdoffset = pos + ifdoffset
        dataoffset += pagedatasize
        if stripoffset2offset is None:
            ifd.seek(stripoffset2value)
            ifd.write(pack(offsetformat, dataoffset))
        else:
            ifd.seek(stripoffset2offset)
            ifd.write(pack(offsetformat, pos + stripoffset2value))
            ifd.seek(stripoffset2value)
            stripoffset = dataoffset
            for size in self._databytecounts:
                ifd.write(pack(offsetformat, stripoffset))
                stripoffset += size
        fh.write(ifd.getvalue())
    self._tags = None
    self._datadtype = None
    self._dataoffset = None
    self._databytecounts = None