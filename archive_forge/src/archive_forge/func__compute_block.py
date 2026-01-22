import io
import struct
import typing
def _compute_block(self, block: typing.List[int], key: typing.List[int]) -> typing.List[int]:
    expanded_block = [block[x] for x in self._e_bit_selection]
    new_block = [int(key[i] != expanded_block[i]) for i in range(48)]
    s_box_perm = []
    s_box_iter = 0
    for i in range(0, 48, 6):
        current_block = new_block[i:i + 6]
        row_bits = [str(current_block[0]), str(current_block[-1])]
        column_bits = [str(x) for x in current_block[1:-1]]
        s_box_row = int(''.join(row_bits), 2)
        s_box_column = int(''.join(column_bits), 2)
        s_box_address = s_box_row * 16 + s_box_column
        s_box_value = self._s_boxes[s_box_iter][s_box_address]
        s_box_iter += 1
        s_box_perm.append(1 if s_box_value & 8 else 0)
        s_box_perm.append(1 if s_box_value & 4 else 0)
        s_box_perm.append(1 if s_box_value & 2 else 0)
        s_box_perm.append(1 if s_box_value & 1 else 0)
    final_block = [s_box_perm[x] for x in self._p]
    return final_block