from .. import osutils
def _insert_long_line(self, line):
    self._flush_insert()
    line_len = len(line)
    for start_index in range(0, line_len, 127):
        next_len = min(127, line_len - start_index)
        self.out_lines.append(bytes([next_len]))
        self.index_lines.append(False)
        self.out_lines.append(line[start_index:start_index + next_len])
        self.index_lines.append(False)