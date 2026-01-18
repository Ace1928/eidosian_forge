import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_set_pixel(self):
    for bpp in (8, 16, 24, 32):
        sf = pygame.Surface((10, 20), 0, bpp)
        sf.fill((0, 0, 0))
        ar = pygame.PixelArray(sf)
        ar.__getitem__(0).__setitem__(0, (0, 255, 0))
        self.assertEqual(ar[0][0], sf.map_rgb((0, 255, 0)))
        ar.__getitem__(1).__setitem__(1, (128, 128, 128))
        self.assertEqual(ar[1][1], sf.map_rgb((128, 128, 128)))
        ar.__getitem__(-1).__setitem__(-1, (128, 128, 128))
        self.assertEqual(ar[9][19], sf.map_rgb((128, 128, 128)))
        ar.__getitem__(-2).__setitem__(-2, (128, 128, 128))
        self.assertEqual(ar[8][-2], sf.map_rgb((128, 128, 128)))