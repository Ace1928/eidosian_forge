from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
def VarData_calculateNumShorts(self, optimize=False):
    count = self.VarRegionCount
    items = self.Item
    bit_lengths = [0] * count
    for item in items:
        bl = [(i + (i < -1)).bit_length() for i in item]
        bit_lengths = [max(*pair) for pair in zip(bl, bit_lengths)]
    byte_lengths = [b + 8 >> 3 if b else 0 for b in bit_lengths]
    longWords = any((b > 2 for b in byte_lengths))
    if optimize:
        mapping = []
        mapping.extend((i for i, b in enumerate(byte_lengths) if b > 2))
        mapping.extend((i for i, b in enumerate(byte_lengths) if b == 2))
        mapping.extend((i for i, b in enumerate(byte_lengths) if b == 1))
        byte_lengths = _reorderItem(byte_lengths, mapping)
        self.VarRegionIndex = _reorderItem(self.VarRegionIndex, mapping)
        self.VarRegionCount = len(self.VarRegionIndex)
        for i in range(len(items)):
            items[i] = _reorderItem(items[i], mapping)
    if longWords:
        self.NumShorts = max((i for i, b in enumerate(byte_lengths) if b > 2), default=-1) + 1
        self.NumShorts |= 32768
    else:
        self.NumShorts = max((i for i, b in enumerate(byte_lengths) if b > 1), default=-1) + 1
    self.VarRegionCount = len(self.VarRegionIndex)
    return self