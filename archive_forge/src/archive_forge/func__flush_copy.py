from .. import osutils
def _flush_copy(self, old_start_linenum, num_lines, out_lines, index_lines):
    if old_start_linenum == 0:
        first_byte = 0
    else:
        first_byte = self.line_offsets[old_start_linenum - 1]
    stop_byte = self.line_offsets[old_start_linenum + num_lines - 1]
    num_bytes = stop_byte - first_byte
    for start_byte in range(first_byte, stop_byte, 64 * 1024):
        num_bytes = min(64 * 1024, stop_byte - start_byte)
        copy_bytes = encode_copy_instruction(start_byte, num_bytes)
        out_lines.append(copy_bytes)
        index_lines.append(False)