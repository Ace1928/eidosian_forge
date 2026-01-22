import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def _check_interface_2D(self, s):
    s_w, s_h = s.get_size()
    s_bytesize = s.get_bytesize()
    s_pitch = s.get_pitch()
    s_pixels = s._pixels_address
    v = s.get_view('2')
    if not IS_PYPY:
        flags = PAI_ALIGNED | PAI_NOTSWAPPED | PAI_WRITEABLE
        if s.get_pitch() == s_w * s_bytesize:
            flags |= PAI_FORTRAN
        inter = ArrayInterface(v)
        self.assertEqual(inter.two, 2)
        self.assertEqual(inter.nd, 2)
        self.assertEqual(inter.typekind, 'u')
        self.assertEqual(inter.itemsize, s_bytesize)
        self.assertEqual(inter.shape[0], s_w)
        self.assertEqual(inter.shape[1], s_h)
        self.assertEqual(inter.strides[0], s_bytesize)
        self.assertEqual(inter.strides[1], s_pitch)
        self.assertEqual(inter.flags, flags)
        self.assertEqual(inter.data, s_pixels)