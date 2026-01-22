import os
import zlib
import time  # noqa
import logging
import numpy as np
class BitArray:
    """Dynamic array of bits that automatically resizes
    with factors of two.
    Append bits using .append() or +=
    You can reverse bits using .reverse()
    """

    def __init__(self, initvalue=None):
        self.data = np.zeros((16,), dtype=np.uint8)
        self._len = 0
        if initvalue is not None:
            self.append(initvalue)

    def __len__(self):
        return self._len

    def __repr__(self):
        return self.data[:self._len].tobytes().decode('ascii')

    def _checkSize(self):
        arraylen = self.data.shape[0]
        if self._len >= arraylen:
            tmp = np.zeros((arraylen * 2,), dtype=np.uint8)
            tmp[:self._len] = self.data[:self._len]
            self.data = tmp

    def __add__(self, value):
        self.append(value)
        return self

    def append(self, bits):
        if isinstance(bits, BitArray):
            bits = str(bits)
        if isinstance(bits, int):
            bits = str(bits)
        if not isinstance(bits, str):
            raise ValueError('Append bits as strings or integers!')
        for bit in bits:
            self.data[self._len] = ord(bit)
            self._len += 1
            self._checkSize()

    def reverse(self):
        """In-place reverse."""
        tmp = self.data[:self._len].copy()
        self.data[:self._len] = tmp[::-1]

    def tobytes(self):
        """Convert to bytes. If necessary,
        zeros are padded to the end (right side).
        """
        bits = str(self)
        nbytes = 0
        while nbytes * 8 < len(bits):
            nbytes += 1
        bits = bits.ljust(nbytes * 8, '0')
        bb = bytes()
        for i in range(nbytes):
            tmp = int(bits[i * 8:(i + 1) * 8], 2)
            bb += int2uint8(tmp)
        return bb