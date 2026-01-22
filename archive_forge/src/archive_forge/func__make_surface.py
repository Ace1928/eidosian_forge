import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def _make_surface(self, bitsize, srcalpha=False, palette=None):
    if palette is None:
        palette = self.test_palette
    flags = 0
    if srcalpha:
        flags |= SRCALPHA
    surf = pygame.Surface(self.surf_size, flags, bitsize)
    if bitsize == 8:
        surf.set_palette([c[:3] for c in palette])
    return surf