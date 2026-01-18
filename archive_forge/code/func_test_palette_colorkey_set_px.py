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
def test_palette_colorkey_set_px(self):
    surf = pygame.image.load(example_path(os.path.join('data', 'alien2.png')))
    key = surf.get_colorkey()
    surf.set_at((0, 0), key)
    self.assertEqual(surf.get_at((0, 0)), key)