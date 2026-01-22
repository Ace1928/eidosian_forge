import os
import numbers
from pathlib import Path
from typing import Union, Set
import numpy as np
from ase.io.jsonio import encode, decode
from ase.utils import plural
class NDArrayReader:

    def __init__(self, fd, shape, dtype, offset, little_endian):
        self.fd = fd
        self.hasfileno = file_has_fileno(fd)
        self.shape = tuple(shape)
        self.dtype = dtype
        self.offset = offset
        self.little_endian = little_endian
        self.ndim = len(self.shape)
        self.itemsize = dtype.itemsize
        self.size = np.prod(self.shape)
        self.nbytes = self.size * self.itemsize
        self.scale = 1.0
        self.length_of_last_dimension = None

    def __len__(self):
        return int(self.shape[0])

    def read(self):
        return self[:]

    def __getitem__(self, i):
        if isinstance(i, numbers.Integral):
            if i < 0:
                i += len(self)
            return self[i:i + 1][0]
        start, stop, step = i.indices(len(self))
        stride = np.prod(self.shape[1:], dtype=int)
        offset = self.offset + start * self.itemsize * stride
        self.fd.seek(offset)
        count = (stop - start) * stride
        if self.hasfileno:
            a = np.fromfile(self.fd, self.dtype, count)
        else:
            a = np.frombuffer(self.fd.read(int(count * self.itemsize)), self.dtype)
        a.shape = (stop - start,) + self.shape[1:]
        if step != 1:
            a = a[::step].copy()
        if self.little_endian != np.little_endian:
            a = a.byteswap(inplace=a.flags.writeable)
        if self.length_of_last_dimension is not None:
            a = a[..., :self.length_of_last_dimension]
        if self.scale != 1.0:
            a *= self.scale
        return a

    def proxy(self, *indices):
        stride = self.size // len(self)
        start = 0
        for i, index in enumerate(indices):
            start += stride * index
            stride //= self.shape[i + 1]
        offset = self.offset + start * self.itemsize
        p = NDArrayReader(self.fd, self.shape[i + 1:], self.dtype, offset, self.little_endian)
        p.scale = self.scale
        return p