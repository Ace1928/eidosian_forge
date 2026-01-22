import re
import itertools
class QRBitBuffer:

    def __init__(self):
        self.buffer = []
        self.length = 0

    def __repr__(self):
        return '.'.join([str(n) for n in self.buffer])

    def get(self, index):
        bufIndex = index // 8
        return self.buffer[bufIndex] >> 7 - index % 8 & 1 == 1

    def put(self, num, length):
        for i in range(length):
            self.putBit(num >> length - i - 1 & 1 == 1)

    def getLengthInBits(self):
        return self.length

    def putBit(self, bit):
        bufIndex = self.length // 8
        if len(self.buffer) <= bufIndex:
            self.buffer.append(0)
        if bit:
            self.buffer[bufIndex] |= 128 >> self.length % 8
        self.length += 1