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
def _check_interface_rgba(self, s, plane):
    s_w, s_h = s.get_size()
    s_bytesize = s.get_bytesize()
    s_pitch = s.get_pitch()
    s_pixels = s._pixels_address
    s_shifts = s.get_shifts()
    s_masks = s.get_masks()
    if not s_masks[plane]:
        return
    alpha_shift = s_shifts[plane]
    offset = alpha_shift // 8
    if not self.lilendian:
        offset = s_bytesize - offset - 1
    v = s.get_view('rgba'[plane])
    if not IS_PYPY:
        inter = ArrayInterface(v)
        flags = PAI_ALIGNED | PAI_NOTSWAPPED | PAI_WRITEABLE
        self.assertEqual(inter.two, 2)
        self.assertEqual(inter.nd, 2)
        self.assertEqual(inter.typekind, 'u')
        self.assertEqual(inter.itemsize, 1)
        self.assertEqual(inter.shape[0], s_w)
        self.assertEqual(inter.shape[1], s_h)
        self.assertEqual(inter.strides[0], s_bytesize)
        self.assertEqual(inter.strides[1], s_pitch)
        self.assertEqual(inter.flags, flags)
        self.assertEqual(inter.data, s_pixels + offset)