import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def _fill_array3d(self, arr):
    palette = self.test_palette
    arr[:5, :6] = palette[1][:3]
    arr[5:, :6] = palette[2][:3]
    arr[:5, 6:] = palette[3][:3]
    arr[5:, 6:] = palette[4][:3]