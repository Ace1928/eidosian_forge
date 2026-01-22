import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def _make_array3d(self, dtype):
    return zeros((self.surf_size[0], self.surf_size[1], 3), dtype)