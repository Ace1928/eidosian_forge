import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def test_pixels2d(self):
    sources = [self._make_surface(8), self._make_surface(16, srcalpha=True), self._make_surface(32, srcalpha=True)]
    for surf in sources:
        self.assertFalse(surf.get_locked())
        arr = pygame.surfarray.pixels2d(surf)
        self.assertTrue(surf.get_locked())
        self._fill_array2d(arr, surf)
        surf.unlock()
        self.assertTrue(surf.get_locked())
        del arr
        self.assertFalse(surf.get_locked())
        self.assertEqual(surf.get_locks(), ())
        self._assert_surface(surf)
    self.assertRaises(ValueError, pygame.surfarray.pixels2d, self._make_surface(24))