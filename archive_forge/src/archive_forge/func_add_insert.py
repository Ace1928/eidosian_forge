from .. import osutils
def add_insert(self, lines):
    if self.cur_insert_lines != []:
        raise AssertionError('self.cur_insert_lines must be empty when adding a new insert')
    for line in lines:
        if len(line) > 127:
            self._insert_long_line(line)
        else:
            next_len = len(line) + self.cur_insert_len
            if next_len > 127:
                self._flush_insert()
                self.cur_insert_lines = [line]
                self.cur_insert_len = len(line)
            else:
                self.cur_insert_lines.append(line)
                self.cur_insert_len = next_len
    self._flush_insert()