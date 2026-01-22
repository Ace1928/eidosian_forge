from __future__ import annotations
import io
import itertools
import logging
import math
import os
import struct
import warnings
from collections.abc import MutableMapping
from fractions import Fraction
from numbers import Number, Rational
from . import ExifTags, Image, ImageFile, ImageOps, ImagePalette, TiffTags
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from .TiffTags import TYPES
class AppendingTiffWriter:
    fieldSizes = [0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8, 4, 2, 4, 8]
    Tags = {273, 288, 324, 519, 520, 521}

    def __init__(self, fn, new=False):
        if hasattr(fn, 'read'):
            self.f = fn
            self.close_fp = False
        else:
            self.name = fn
            self.close_fp = True
            try:
                self.f = open(fn, 'w+b' if new else 'r+b')
            except OSError:
                self.f = open(fn, 'w+b')
        self.beginning = self.f.tell()
        self.setup()

    def setup(self):
        self.f.seek(self.beginning, os.SEEK_SET)
        self.whereToWriteNewIFDOffset = None
        self.offsetOfNewPage = 0
        self.IIMM = iimm = self.f.read(4)
        if not iimm:
            self.isFirst = True
            return
        self.isFirst = False
        if iimm == b'II*\x00':
            self.setEndian('<')
        elif iimm == b'MM\x00*':
            self.setEndian('>')
        else:
            msg = 'Invalid TIFF file header'
            raise RuntimeError(msg)
        self.skipIFDs()
        self.goToEnd()

    def finalize(self):
        if self.isFirst:
            return
        self.f.seek(self.offsetOfNewPage)
        iimm = self.f.read(4)
        if not iimm:
            return
        if iimm != self.IIMM:
            msg = "IIMM of new page doesn't match IIMM of first page"
            raise RuntimeError(msg)
        ifd_offset = self.readLong()
        ifd_offset += self.offsetOfNewPage
        self.f.seek(self.whereToWriteNewIFDOffset)
        self.writeLong(ifd_offset)
        self.f.seek(ifd_offset)
        self.fixIFD()

    def newFrame(self):
        self.finalize()
        self.setup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.close_fp:
            self.close()
        return False

    def tell(self):
        return self.f.tell() - self.offsetOfNewPage

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == os.SEEK_SET:
            offset += self.offsetOfNewPage
        self.f.seek(offset, whence)
        return self.tell()

    def goToEnd(self):
        self.f.seek(0, os.SEEK_END)
        pos = self.f.tell()
        pad_bytes = 16 - pos % 16
        if 0 < pad_bytes < 16:
            self.f.write(bytes(pad_bytes))
        self.offsetOfNewPage = self.f.tell()

    def setEndian(self, endian):
        self.endian = endian
        self.longFmt = self.endian + 'L'
        self.shortFmt = self.endian + 'H'
        self.tagFormat = self.endian + 'HHL'

    def skipIFDs(self):
        while True:
            ifd_offset = self.readLong()
            if ifd_offset == 0:
                self.whereToWriteNewIFDOffset = self.f.tell() - 4
                break
            self.f.seek(ifd_offset)
            num_tags = self.readShort()
            self.f.seek(num_tags * 12, os.SEEK_CUR)

    def write(self, data):
        return self.f.write(data)

    def readShort(self):
        value, = struct.unpack(self.shortFmt, self.f.read(2))
        return value

    def readLong(self):
        value, = struct.unpack(self.longFmt, self.f.read(4))
        return value

    def rewriteLastShortToLong(self, value):
        self.f.seek(-2, os.SEEK_CUR)
        bytes_written = self.f.write(struct.pack(self.longFmt, value))
        if bytes_written is not None and bytes_written != 4:
            msg = f'wrote only {bytes_written} bytes but wanted 4'
            raise RuntimeError(msg)

    def rewriteLastShort(self, value):
        self.f.seek(-2, os.SEEK_CUR)
        bytes_written = self.f.write(struct.pack(self.shortFmt, value))
        if bytes_written is not None and bytes_written != 2:
            msg = f'wrote only {bytes_written} bytes but wanted 2'
            raise RuntimeError(msg)

    def rewriteLastLong(self, value):
        self.f.seek(-4, os.SEEK_CUR)
        bytes_written = self.f.write(struct.pack(self.longFmt, value))
        if bytes_written is not None and bytes_written != 4:
            msg = f'wrote only {bytes_written} bytes but wanted 4'
            raise RuntimeError(msg)

    def writeShort(self, value):
        bytes_written = self.f.write(struct.pack(self.shortFmt, value))
        if bytes_written is not None and bytes_written != 2:
            msg = f'wrote only {bytes_written} bytes but wanted 2'
            raise RuntimeError(msg)

    def writeLong(self, value):
        bytes_written = self.f.write(struct.pack(self.longFmt, value))
        if bytes_written is not None and bytes_written != 4:
            msg = f'wrote only {bytes_written} bytes but wanted 4'
            raise RuntimeError(msg)

    def close(self):
        self.finalize()
        self.f.close()

    def fixIFD(self):
        num_tags = self.readShort()
        for i in range(num_tags):
            tag, field_type, count = struct.unpack(self.tagFormat, self.f.read(8))
            field_size = self.fieldSizes[field_type]
            total_size = field_size * count
            is_local = total_size <= 4
            if not is_local:
                offset = self.readLong()
                offset += self.offsetOfNewPage
                self.rewriteLastLong(offset)
            if tag in self.Tags:
                cur_pos = self.f.tell()
                if is_local:
                    self.fixOffsets(count, isShort=field_size == 2, isLong=field_size == 4)
                    self.f.seek(cur_pos + 4)
                else:
                    self.f.seek(offset)
                    self.fixOffsets(count, isShort=field_size == 2, isLong=field_size == 4)
                    self.f.seek(cur_pos)
                offset = cur_pos = None
            elif is_local:
                self.f.seek(4, os.SEEK_CUR)

    def fixOffsets(self, count, isShort=False, isLong=False):
        if not isShort and (not isLong):
            msg = 'offset is neither short nor long'
            raise RuntimeError(msg)
        for i in range(count):
            offset = self.readShort() if isShort else self.readLong()
            offset += self.offsetOfNewPage
            if isShort and offset >= 65536:
                if count != 1:
                    msg = 'not implemented'
                    raise RuntimeError(msg)
                self.rewriteLastShortToLong(offset)
                self.f.seek(-10, os.SEEK_CUR)
                self.writeShort(TiffTags.LONG)
                self.f.seek(8, os.SEEK_CUR)
            elif isShort:
                self.rewriteLastShort(offset)
            else:
                self.rewriteLastLong(offset)