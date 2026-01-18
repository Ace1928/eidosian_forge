import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def test_pixel_alpha(self):
    bitsizes = [16, 32]
    for bitsize in bitsizes:
        surf = self._make_surface(bitsize, srcalpha=True)
        comp = self._make_surface(bitsize, srcalpha=True)
        comp.blit(surf, (3, 0))
        surf.blit(surf, (3, 0))
        self._assert_same(surf, comp)