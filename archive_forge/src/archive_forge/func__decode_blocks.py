from __future__ import annotations
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
from ._binary import o32le as o32
def _decode_blocks(self, maxval):
    data = bytearray()
    max_len = 10
    out_byte_count = 4 if self.mode == 'I' else 1
    out_max = 65535 if self.mode == 'I' else 255
    bands = Image.getmodebands(self.mode)
    total_bytes = self.state.xsize * self.state.ysize * bands * out_byte_count
    half_token = False
    while len(data) != total_bytes:
        block = self._read_block()
        if not block:
            if half_token:
                block = bytearray(b' ')
            else:
                break
        block = self._ignore_comments(block)
        if half_token:
            block = half_token + block
            half_token = False
        tokens = block.split()
        if block and (not block[-1:].isspace()):
            half_token = tokens.pop()
            if len(half_token) > max_len:
                msg = b'Token too long found in data: %s' % half_token[:max_len + 1]
                raise ValueError(msg)
        for token in tokens:
            if len(token) > max_len:
                msg = b'Token too long found in data: %s' % token[:max_len + 1]
                raise ValueError(msg)
            value = int(token)
            if value > maxval:
                msg = f'Channel value too large for this mode: {value}'
                raise ValueError(msg)
            value = round(value / maxval * out_max)
            data += o32(value) if self.mode == 'I' else o8(value)
            if len(data) == total_bytes:
                break
    return data