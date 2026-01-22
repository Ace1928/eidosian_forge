import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def _make_src_array3d(self, dtype):
    arr = self._make_array3d(dtype)
    self._fill_array3d(arr)
    return arr