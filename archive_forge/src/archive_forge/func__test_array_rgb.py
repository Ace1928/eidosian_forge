import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def _test_array_rgb(self, operation, mask_posn):
    method_name = 'array_' + operation
    array_rgb = getattr(pygame.surfarray, method_name)
    palette = [(0, 0, 0, 255), (5, 13, 23, 255), (29, 31, 37, 255), (131, 157, 167, 255), (179, 191, 251, 255)]
    plane = [c[mask_posn] for c in palette]
    targets = [self._make_src_surface(24, palette=palette), self._make_src_surface(32, palette=palette), self._make_src_surface(32, palette=palette, srcalpha=True)]
    for surf in targets:
        self.assertFalse(surf.get_locked())
        for (x, y), i in self.test_points:
            surf.fill(palette[i])
            arr = array_rgb(surf)
            self.assertEqual(arr[x, y], plane[i])
            surf.fill((100, 100, 100, 250))
            self.assertEqual(arr[x, y], plane[i])
            self.assertFalse(surf.get_locked())
            del arr