import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def test_surface_to_array_newbuf(self):
    array = self.Array2D(range(0, 15))
    self.assertNotEqual(array.content[0], self.surface.get_at_mapped((0, 0)))
    surface_to_array(array, self.surface)
    self.assertCopy2D(self.surface, array)