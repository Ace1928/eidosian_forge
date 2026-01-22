import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def _fill_array2d(self, arr, surf):
    palette = self.test_palette
    arr[:5, :6] = surf.map_rgb(palette[1])
    arr[5:, :6] = surf.map_rgb(palette[2])
    arr[:5, 6:] = surf.map_rgb(palette[3])
    arr[5:, 6:] = surf.map_rgb(palette[4])