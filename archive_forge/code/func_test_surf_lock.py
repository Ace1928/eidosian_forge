import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def test_surf_lock(self):
    sf = pygame.Surface((5, 5), 0, 32)
    for atype in pygame.surfarray.get_arraytypes():
        pygame.surfarray.use_arraytype(atype)
        ar = pygame.surfarray.pixels2d(sf)
        self.assertTrue(sf.get_locked())
        sf.unlock()
        self.assertTrue(sf.get_locked())
        del ar
        self.assertFalse(sf.get_locked())
        self.assertEqual(sf.get_locks(), ())