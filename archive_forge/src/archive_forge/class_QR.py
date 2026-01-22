import re
import itertools
class QR:
    valid = None
    bits = None
    group = 0

    def __init__(self, data):
        if self.valid and (not self.valid(data)):
            raise ValueError
        self.data = data

    def __len__(self):
        return len(self.data)

    @property
    def bitlength(self):
        if self.bits is None:
            return 0
        q, r = divmod(len(self), len(self.bits))
        return q * sum(self.bits) + sum(self.bits[:r])

    def getLengthBits(self, ver):
        if 0 < ver < 10:
            return self.lengthbits[0]
        elif ver < 27:
            return self.lengthbits[1]
        elif ver < 41:
            return self.lengthbits[2]
        raise ValueError('Unknown version: ' + ver)

    def getLength(self):
        return len(self.data)

    def __repr__(self):
        return repr(self.data)

    def write_header(self, buffer, version):
        buffer.put(self.mode, 4)
        lenbits = self.getLengthBits(version)
        if lenbits:
            buffer.put(len(self.data), lenbits)

    def write(self, buffer, version):
        self.write_header(buffer, version)
        for g in zip_longest(*[iter(self.data)] * self.group):
            bits = 0
            n = 0
            for i in range(self.group):
                if g[i] is not None:
                    n *= len(self.chars)
                    n += self.chars.index(g[i])
                    bits += self.bits[i]
            buffer.put(n, bits)