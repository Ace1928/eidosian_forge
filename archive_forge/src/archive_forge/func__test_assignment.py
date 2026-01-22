import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def _test_assignment(self, sf, ar, ar_size, ar_strides, ar_offset):
    self.assertEqual(ar.shape, ar_size)
    ar_w, ar_h = ar_size
    ar_xstride, ar_ystride = ar_strides
    sf_w, sf_h = sf.get_size()
    black = pygame.Color('black')
    color = pygame.Color(0, 0, 12)
    pxcolor = sf.map_rgb(color)
    sf.fill(black)
    for ar_x, ar_y in [(0, 0), (0, ar_h - 4), (ar_w - 3, 0), (0, ar_h - 1), (ar_w - 1, 0), (ar_w - 1, ar_h - 1)]:
        sf_offset = ar_offset + ar_x * ar_xstride + ar_y * ar_ystride
        sf_y = sf_offset // sf_w
        sf_x = sf_offset - sf_y * sf_w
        sf_posn = (sf_x, sf_y)
        sf_pix = sf.get_at(sf_posn)
        self.assertEqual(sf_pix, black, 'at pixarr posn (%i, %i) (surf posn (%i, %i)): %s != %s' % (ar_x, ar_y, sf_x, sf_y, sf_pix, black))
        ar[ar_x, ar_y] = pxcolor
        sf_pix = sf.get_at(sf_posn)
        self.assertEqual(sf_pix, color, 'at pixarr posn (%i, %i) (surf posn (%i, %i)): %s != %s' % (ar_x, ar_y, sf_x, sf_y, sf_pix, color))