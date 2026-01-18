import re
import itertools
def write_header(self, buffer, version):
    buffer.put(self.mode, 4)
    buffer.put(1, 4)
    lenbits = self.getLengthBits(version)
    if lenbits:
        buffer.put(len(self.data), lenbits)