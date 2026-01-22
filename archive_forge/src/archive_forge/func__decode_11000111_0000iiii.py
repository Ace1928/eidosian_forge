from collections import namedtuple
def _decode_11000111_0000iiii(self):
    self._index += 1
    op1 = self._bytecode_array[self._index]
    self._index += 1
    if op1 & 240 != 0 or op1 == 0:
        return 'spare'
    else:
        return 'pop %s' % self._print_registers(op1 & 15, 'wCGR')