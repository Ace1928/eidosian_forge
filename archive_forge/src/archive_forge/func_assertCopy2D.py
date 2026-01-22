import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def assertCopy2D(self, surface, array):
    for x in range(0, 3):
        for y in range(0, 5):
            self.assertEqual(surface.get_at_mapped((x, y)), array[x, y])