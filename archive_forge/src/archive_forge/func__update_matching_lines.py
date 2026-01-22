from .. import osutils
def _update_matching_lines(self, new_lines, index):
    matches = self._matching_lines
    start_idx = len(self.lines)
    if len(new_lines) != len(index):
        raise AssertionError("The number of lines to be indexed does not match the index/don't index flags: %d != %d" % (len(new_lines), len(index)))
    for idx, do_index in enumerate(index):
        if not do_index:
            continue
        line = new_lines[idx]
        try:
            matches[line].add(start_idx + idx)
        except KeyError:
            matches[line] = {start_idx + idx}