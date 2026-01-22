import array
import binascii
import io
import os
import tempfile
import unittest
import glob
import pathlib
from pygame.tests.test_utils import example_path, png, tostring
import pygame, pygame.image, pygame.pkgdata
def _assertSurfaceEqual(self, surf_a, surf_b, msg=None):
    a_width, a_height = (surf_a.get_width(), surf_a.get_height())
    self.assertEqual(a_width, surf_b.get_width(), msg)
    self.assertEqual(a_height, surf_b.get_height(), msg)
    self.assertEqual(surf_a.get_size(), surf_b.get_size(), msg)
    self.assertEqual(surf_a.get_rect(), surf_b.get_rect(), msg)
    self.assertEqual(surf_a.get_colorkey(), surf_b.get_colorkey(), msg)
    self.assertEqual(surf_a.get_alpha(), surf_b.get_alpha(), msg)
    self.assertEqual(surf_a.get_flags(), surf_b.get_flags(), msg)
    self.assertEqual(surf_a.get_bitsize(), surf_b.get_bitsize(), msg)
    self.assertEqual(surf_a.get_bytesize(), surf_b.get_bytesize(), msg)
    surf_a_get_at = surf_a.get_at
    surf_b_get_at = surf_b.get_at
    for y in range(a_height):
        for x in range(a_width):
            self.assertEqual(surf_a_get_at((x, y)), surf_b_get_at((x, y)), '%s (pixel: %d, %d)' % (msg, x, y))