import os
import mmap
import struct
import codecs
def _add_kernings_format0(self, kernings, offset):
    header = _read_kern_subtable_format0(self._data, offset)
    kerning_pairs = _read_kern_subtable_format0Pair.array(self._data, offset + header.size, header.n_pairs)
    for pair in kerning_pairs:
        if (pair.left, pair.right) in kernings:
            kernings[pair.left, pair.right] += pair.value / float(self.header.units_per_em)
        else:
            kernings[pair.left, pair.right] = pair.value / float(self.header.units_per_em)