import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def _test_pixels_rgb(self, operation, mask_posn):
    method_name = 'pixels_' + operation
    pixels_rgb = getattr(pygame.surfarray, method_name)
    palette = [(0, 0, 0, 255), (5, 13, 23, 255), (29, 31, 37, 255), (131, 157, 167, 255), (179, 191, 251, 255)]
    plane = [c[mask_posn] for c in palette]
    surf24 = self._make_src_surface(24, srcalpha=False, palette=palette)
    surf32 = self._make_src_surface(32, srcalpha=False, palette=palette)
    surf32a = self._make_src_surface(32, srcalpha=True, palette=palette)
    for surf in [surf24, surf32, surf32a]:
        self.assertFalse(surf.get_locked())
        arr = pixels_rgb(surf)
        self.assertTrue(surf.get_locked())
        surf.unlock()
        self.assertTrue(surf.get_locked())
        for (x, y), i in self.test_points:
            self.assertEqual(arr[x, y], plane[i])
        del arr
        self.assertFalse(surf.get_locked())
        self.assertEqual(surf.get_locks(), ())
    targets = [(8, False), (16, False), (16, True)]
    for bitsize, srcalpha in targets:
        self.assertRaises(ValueError, pixels_rgb, self._make_surface(bitsize, srcalpha))