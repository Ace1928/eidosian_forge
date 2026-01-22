import sys
import unittest
import platform
import pygame
class ExporterBase:

    def __init__(self, shape, typechar, itemsize):
        import ctypes
        ndim = len(shape)
        self.ndim = ndim
        self.shape = tuple(shape)
        array_len = 1
        for d in shape:
            array_len *= d
        self.size = itemsize * array_len
        self.parent = ctypes.create_string_buffer(self.size)
        self.itemsize = itemsize
        strides = [itemsize] * ndim
        for i in range(ndim - 1, 0, -1):
            strides[i - 1] = strides[i] * shape[i]
        self.strides = tuple(strides)
        self.data = (ctypes.addressof(self.parent), False)
        if self.itemsize == 1:
            byteorder = '|'
        elif sys.byteorder == 'big':
            byteorder = '>'
        else:
            byteorder = '<'
        self.typestr = byteorder + typechar + str(self.itemsize)