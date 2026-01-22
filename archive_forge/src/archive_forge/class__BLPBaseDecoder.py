from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
class _BLPBaseDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        try:
            self._read_blp_header()
            self._load()
        except struct.error as e:
            msg = 'Truncated BLP file'
            raise OSError(msg) from e
        return (-1, 0)

    def _read_blp_header(self):
        self.fd.seek(4)
        self._blp_compression, = struct.unpack('<i', self._safe_read(4))
        self._blp_encoding, = struct.unpack('<b', self._safe_read(1))
        self._blp_alpha_depth, = struct.unpack('<b', self._safe_read(1))
        self._blp_alpha_encoding, = struct.unpack('<b', self._safe_read(1))
        self.fd.seek(1, os.SEEK_CUR)
        self.size = struct.unpack('<II', self._safe_read(8))
        if isinstance(self, BLP1Decoder):
            self._blp_encoding, = struct.unpack('<i', self._safe_read(4))
            self.fd.seek(4, os.SEEK_CUR)
        self._blp_offsets = struct.unpack('<16I', self._safe_read(16 * 4))
        self._blp_lengths = struct.unpack('<16I', self._safe_read(16 * 4))

    def _safe_read(self, length):
        return ImageFile._safe_read(self.fd, length)

    def _read_palette(self):
        ret = []
        for i in range(256):
            try:
                b, g, r, a = struct.unpack('<4B', self._safe_read(4))
            except struct.error:
                break
            ret.append((b, g, r, a))
        return ret

    def _read_bgra(self, palette):
        data = bytearray()
        _data = BytesIO(self._safe_read(self._blp_lengths[0]))
        while True:
            try:
                offset, = struct.unpack('<B', _data.read(1))
            except struct.error:
                break
            b, g, r, a = palette[offset]
            d = (r, g, b)
            if self._blp_alpha_depth:
                d += (a,)
            data.extend(d)
        return data