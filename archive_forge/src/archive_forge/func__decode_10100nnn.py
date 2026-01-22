from collections import namedtuple
def _decode_10100nnn(self):
    opcode = self._bytecode_array[self._index]
    self._index += 1
    return 'pop %s' % self._printGPR(self._calculate_range(4, opcode & 7))