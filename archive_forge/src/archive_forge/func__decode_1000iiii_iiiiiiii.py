from collections import namedtuple
def _decode_1000iiii_iiiiiiii(self):
    op0 = self._bytecode_array[self._index]
    self._index += 1
    op1 = self._bytecode_array[self._index]
    self._index += 1
    gpr_mask = op1 << 4 | (op0 & 15) << 12
    if gpr_mask == 0:
        return 'refuse to unwind'
    else:
        return 'pop %s' % self._printGPR(gpr_mask)