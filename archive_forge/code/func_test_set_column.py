import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_set_column(self):
    for bpp in (8, 16, 24, 32):
        sf = pygame.Surface((6, 8), 0, bpp)
        sf.fill((0, 0, 0))
        ar = pygame.PixelArray(sf)
        sf2 = pygame.Surface((6, 8), 0, bpp)
        sf2.fill((0, 255, 255))
        ar2 = pygame.PixelArray(sf2)
        ar.__setitem__(2, (128, 128, 128))
        self.assertEqual(ar[2][0], sf.map_rgb((128, 128, 128)))
        self.assertEqual(ar[2][1], sf.map_rgb((128, 128, 128)))
        ar.__setitem__(-1, (0, 255, 255))
        self.assertEqual(ar[5][0], sf.map_rgb((0, 255, 255)))
        self.assertEqual(ar[-1][1], sf.map_rgb((0, 255, 255)))
        ar.__setitem__(-2, (255, 255, 0))
        self.assertEqual(ar[4][0], sf.map_rgb((255, 255, 0)))
        self.assertEqual(ar[-2][1], sf.map_rgb((255, 255, 0)))
        ar.__setitem__(0, [(255, 255, 255)] * 8)
        self.assertEqual(ar[0][0], sf.map_rgb((255, 255, 255)))
        self.assertEqual(ar[0][1], sf.map_rgb((255, 255, 255)))
        self.assertRaises(ValueError, ar.__setitem__, 1, ((204, 0, 204), (17, 17, 17), (204, 0, 204), (17, 17, 17), (204, 0, 204), (17, 17, 17), (204, 0, 204), (17, 17, 17)))
        ar.__setitem__(1, ar2.__getitem__(3))
        self.assertEqual(ar[1][0], sf.map_rgb((0, 255, 255)))
        self.assertEqual(ar[1][1], sf.map_rgb((0, 255, 255)))