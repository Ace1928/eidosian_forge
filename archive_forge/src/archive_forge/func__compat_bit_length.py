from __future__ import unicode_literals
import itertools
import struct
def _compat_bit_length(i):
    for res in itertools.count():
        if i >> res == 0:
            return res