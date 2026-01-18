import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_pixels_field(self):
    for bpp in [1, 2, 3, 4]:
        sf = pygame.Surface((11, 7), 0, bpp * 8)
        ar = pygame.PixelArray(sf)
        ar2 = ar[1:, :]
        self.assertEqual(ar2._pixels_address - ar._pixels_address, ar.itemsize)
        ar2 = ar[:, 1:]
        self.assertEqual(ar2._pixels_address - ar._pixels_address, ar.strides[1])
        ar2 = ar[::-1, :]
        self.assertEqual(ar2._pixels_address - ar._pixels_address, (ar.shape[0] - 1) * ar.itemsize)
        ar2 = ar[::-2, :]
        self.assertEqual(ar2._pixels_address - ar._pixels_address, (ar.shape[0] - 1) * ar.itemsize)
        ar2 = ar[:, ::-1]
        self.assertEqual(ar2._pixels_address - ar._pixels_address, (ar.shape[1] - 1) * ar.strides[1])
        ar3 = ar2[::-1, :]
        self.assertEqual(ar3._pixels_address - ar._pixels_address, (ar.shape[0] - 1) * ar.strides[0] + (ar.shape[1] - 1) * ar.strides[1])
        ar2 = ar[:, ::-2]
        self.assertEqual(ar2._pixels_address - ar._pixels_address, (ar.shape[1] - 1) * ar.strides[1])
        ar2 = ar[2:, 3:]
        self.assertEqual(ar2._pixels_address - ar._pixels_address, ar.strides[0] * 2 + ar.strides[1] * 3)
        ar2 = ar[2::2, 3::4]
        self.assertEqual(ar2._pixels_address - ar._pixels_address, ar.strides[0] * 2 + ar.strides[1] * 3)
        ar2 = ar[9:2:-1, :]
        self.assertEqual(ar2._pixels_address - ar._pixels_address, ar.strides[0] * 9)
        ar2 = ar[:, 5:2:-1]
        self.assertEqual(ar2._pixels_address - ar._pixels_address, ar.strides[1] * 5)