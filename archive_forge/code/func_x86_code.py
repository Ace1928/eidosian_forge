import struct
from typing import Union
def x86_code(self) -> int:
    """
        The code algorithm from liblzma/simple/x86.c
        It is slightly different from LZMA-SDK's bra86.c
        :return: buffer position
        """
    size: int = len(self.buffer)
    if size < 5:
        return 0
    if self.current_position - self.prev_pos > 5:
        self.prev_pos = self.current_position - 5
    view = memoryview(self.buffer)
    limit: int = size - 5
    buffer_pos: int = 0
    pos1: int = 0
    pos2: int = 0
    while buffer_pos <= limit:
        if pos1 >= 0:
            pos1 = self.buffer.find(233, buffer_pos, limit)
        if pos2 >= 0:
            pos2 = self.buffer.find(232, buffer_pos, limit)
        if pos1 < 0 and pos2 < 0:
            buffer_pos = limit + 1
            break
        elif pos1 < 0:
            buffer_pos = pos2
        elif pos2 < 0:
            buffer_pos = pos1
        else:
            buffer_pos = min(pos1, pos2)
        offset = self.current_position + buffer_pos - self.prev_pos
        self.prev_pos = self.current_position + buffer_pos
        if offset > 5:
            self.prev_mask = 0
        else:
            for i in range(offset):
                self.prev_mask &= 119
                self.prev_mask <<= 1
        if view[buffer_pos + 4] in [0, 255] and self.prev_mask >> 1 in self._mask_to_allowed_number:
            jump_target = self.buffer[buffer_pos + 1:buffer_pos + 5]
            src = struct.unpack('<L', jump_target)[0]
            distance = self.current_position + buffer_pos + 5
            idx = self._mask_to_bit_number[self.prev_mask >> 1]
            while True:
                if self.is_encoder:
                    dest = src + distance & 4294967295
                else:
                    dest = src - distance & 4294967295
                if self.prev_mask == 0:
                    break
                b = 255 & dest >> 24 - idx * 8
                if not (b == 0 or b == 255):
                    break
                src = dest ^ (1 << 32 - idx * 8) - 1 & 4294967295
            write_view = view[buffer_pos + 1:buffer_pos + 5]
            write_view[0:3] = (dest & 16777215).to_bytes(3, 'little')
            write_view[3:4] = [b'\x00', b'\xff'][dest >> 24 & 1]
            buffer_pos += 5
            self.prev_mask = 0
        else:
            buffer_pos += 1
            self.prev_mask |= 1
            if self.buffer[buffer_pos + 3] in [0, 255]:
                self.prev_mask |= 16
    self.current_position += buffer_pos
    return buffer_pos