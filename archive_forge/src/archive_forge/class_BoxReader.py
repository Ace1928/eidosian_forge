from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
class BoxReader:
    """
    A small helper class to read fields stored in JPEG2000 header boxes
    and to easily step into and read sub-boxes.
    """

    def __init__(self, fp, length=-1):
        self.fp = fp
        self.has_length = length >= 0
        self.length = length
        self.remaining_in_box = -1

    def _can_read(self, num_bytes):
        if self.has_length and self.fp.tell() + num_bytes > self.length:
            return False
        if self.remaining_in_box >= 0:
            return num_bytes <= self.remaining_in_box
        else:
            return True

    def _read_bytes(self, num_bytes):
        if not self._can_read(num_bytes):
            msg = 'Not enough data in header'
            raise SyntaxError(msg)
        data = self.fp.read(num_bytes)
        if len(data) < num_bytes:
            msg = f'Expected to read {num_bytes} bytes but only got {len(data)}.'
            raise OSError(msg)
        if self.remaining_in_box > 0:
            self.remaining_in_box -= num_bytes
        return data

    def read_fields(self, field_format):
        size = struct.calcsize(field_format)
        data = self._read_bytes(size)
        return struct.unpack(field_format, data)

    def read_boxes(self):
        size = self.remaining_in_box
        data = self._read_bytes(size)
        return BoxReader(io.BytesIO(data), size)

    def has_next_box(self):
        if self.has_length:
            return self.fp.tell() + self.remaining_in_box < self.length
        else:
            return True

    def next_box_type(self):
        if self.remaining_in_box > 0:
            self.fp.seek(self.remaining_in_box, os.SEEK_CUR)
        self.remaining_in_box = -1
        lbox, tbox = self.read_fields('>I4s')
        if lbox == 1:
            lbox = self.read_fields('>Q')[0]
            hlen = 16
        else:
            hlen = 8
        if lbox < hlen or not self._can_read(lbox - hlen):
            msg = 'Invalid header length'
            raise SyntaxError(msg)
        self.remaining_in_box = lbox - hlen
        return tbox