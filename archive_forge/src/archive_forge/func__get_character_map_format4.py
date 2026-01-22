import os
import mmap
import struct
import codecs
def _get_character_map_format4(self, offset):
    header = _read_cmap_format4Header(self._data, offset)
    seg_count = header.seg_count_x2 // 2
    array_size = struct.calcsize(f'>{seg_count}H')
    end_count = self._read_array(f'>{seg_count}H', offset + header.size)
    start_count = self._read_array(f'>{seg_count}H', offset + header.size + array_size + 2)
    id_delta = self._read_array(f'>{seg_count}H', offset + header.size + array_size + 2 + array_size)
    id_range_offset_address = offset + header.size + array_size + 2 + array_size + array_size
    id_range_offset = self._read_array(f'>{seg_count}H', id_range_offset_address)
    character_map = {}
    for i in range(0, seg_count):
        if id_range_offset[i] != 0:
            if id_range_offset[i] == 65535:
                continue
            for c in range(start_count[i], end_count[i] + 1):
                addr = id_range_offset[i] + 2 * (c - start_count[i]) + id_range_offset_address + 2 * i
                g = struct.unpack('>H', self._data[addr:addr + 2])[0]
                if g != 0:
                    character_map[chr(c)] = (g + id_delta[i]) % 65536
        else:
            for c in range(start_count[i], end_count[i] + 1):
                g = (c + id_delta[i]) % 65536
                if g != 0:
                    character_map[chr(c)] = g
    return character_map