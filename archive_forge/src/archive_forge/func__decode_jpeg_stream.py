from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
def _decode_jpeg_stream(self):
    from .JpegImagePlugin import JpegImageFile
    jpeg_header_size, = struct.unpack('<I', self._safe_read(4))
    jpeg_header = self._safe_read(jpeg_header_size)
    self._safe_read(self._blp_offsets[0] - self.fd.tell())
    data = self._safe_read(self._blp_lengths[0])
    data = jpeg_header + data
    data = BytesIO(data)
    image = JpegImageFile(data)
    Image._decompression_bomb_check(image.size)
    if image.mode == 'CMYK':
        decoder_name, extents, offset, args = image.tile[0]
        image.tile = [(decoder_name, extents, offset, (args[0], 'CMYK'))]
    r, g, b = image.convert('RGB').split()
    image = Image.merge('RGB', (b, g, r))
    self.set_as_raw(image.tobytes())