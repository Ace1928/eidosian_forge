from __future__ import annotations
import os
from . import Image, ImageFile
from ._binary import i32be as i32
from ._binary import o8
class QoiDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def _add_to_previous_pixels(self, value):
        self._previous_pixel = value
        r, g, b, a = value
        hash_value = (r * 3 + g * 5 + b * 7 + a * 11) % 64
        self._previously_seen_pixels[hash_value] = value

    def decode(self, buffer):
        self._previously_seen_pixels = {}
        self._previous_pixel = None
        self._add_to_previous_pixels(b''.join((o8(i) for i in (0, 0, 0, 255))))
        data = bytearray()
        bands = Image.getmodebands(self.mode)
        while len(data) < self.state.xsize * self.state.ysize * bands:
            byte = self.fd.read(1)[0]
            if byte == 254:
                value = self.fd.read(3) + self._previous_pixel[3:]
            elif byte == 255:
                value = self.fd.read(4)
            else:
                op = byte >> 6
                if op == 0:
                    op_index = byte & 63
                    value = self._previously_seen_pixels.get(op_index, (0, 0, 0, 0))
                elif op == 1:
                    value = ((self._previous_pixel[0] + ((byte & 48) >> 4) - 2) % 256, (self._previous_pixel[1] + ((byte & 12) >> 2) - 2) % 256, (self._previous_pixel[2] + (byte & 3) - 2) % 256)
                    value += (self._previous_pixel[3],)
                elif op == 2:
                    second_byte = self.fd.read(1)[0]
                    diff_green = (byte & 63) - 32
                    diff_red = ((second_byte & 240) >> 4) - 8
                    diff_blue = (second_byte & 15) - 8
                    value = tuple(((self._previous_pixel[i] + diff_green + diff) % 256 for i, diff in enumerate((diff_red, 0, diff_blue))))
                    value += (self._previous_pixel[3],)
                elif op == 3:
                    run_length = (byte & 63) + 1
                    value = self._previous_pixel
                    if bands == 3:
                        value = value[:3]
                    data += value * run_length
                    continue
                value = b''.join((o8(i) for i in value))
            self._add_to_previous_pixels(value)
            if bands == 3:
                value = value[:3]
            data += value
        self.set_as_raw(bytes(data))
        return (-1, 0)