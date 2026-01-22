from functools import total_ordering
def _set_single_rebuild(self, index, value):
    self._set_slice(slice(index, index + 1, 1), [value])