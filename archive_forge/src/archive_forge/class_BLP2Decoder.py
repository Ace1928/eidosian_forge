from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
class BLP2Decoder(_BLPBaseDecoder):

    def _load(self):
        palette = self._read_palette()
        self.fd.seek(self._blp_offsets[0])
        if self._blp_compression == 1:
            if self._blp_encoding == Encoding.UNCOMPRESSED:
                data = self._read_bgra(palette)
            elif self._blp_encoding == Encoding.DXT:
                data = bytearray()
                if self._blp_alpha_encoding == AlphaEncoding.DXT1:
                    linesize = (self.size[0] + 3) // 4 * 8
                    for yb in range((self.size[1] + 3) // 4):
                        for d in decode_dxt1(self._safe_read(linesize), alpha=bool(self._blp_alpha_depth)):
                            data += d
                elif self._blp_alpha_encoding == AlphaEncoding.DXT3:
                    linesize = (self.size[0] + 3) // 4 * 16
                    for yb in range((self.size[1] + 3) // 4):
                        for d in decode_dxt3(self._safe_read(linesize)):
                            data += d
                elif self._blp_alpha_encoding == AlphaEncoding.DXT5:
                    linesize = (self.size[0] + 3) // 4 * 16
                    for yb in range((self.size[1] + 3) // 4):
                        for d in decode_dxt5(self._safe_read(linesize)):
                            data += d
                else:
                    msg = f'Unsupported alpha encoding {repr(self._blp_alpha_encoding)}'
                    raise BLPFormatError(msg)
            else:
                msg = f'Unknown BLP encoding {repr(self._blp_encoding)}'
                raise BLPFormatError(msg)
        else:
            msg = f'Unknown BLP compression {repr(self._blp_compression)}'
            raise BLPFormatError(msg)
        self.set_as_raw(bytes(data))