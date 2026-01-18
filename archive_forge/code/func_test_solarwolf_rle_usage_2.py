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
def test_solarwolf_rle_usage_2(self):
    """Test for RLE status after setting alpha"""
    pygame.display.init()
    try:
        pygame.display.set_mode((640, 480), depth=32)
        blit_to_surf = pygame.Surface((100, 100))
        image = pygame.image.load(example_path(os.path.join('data', 'alien1.png')))
        image = image.convert()
        orig_colorkey = image.get_colorkey()
        image.set_colorkey(orig_colorkey, RLEACCEL)
        self.assertTrue(image.get_flags() & pygame.RLEACCELOK)
        self.assertTrue(not image.get_flags() & pygame.RLEACCEL)
        blit_to_surf.blit(image, (0, 0))
        self.assertTrue(image.get_flags() & pygame.RLEACCELOK)
        self.assertTrue(image.get_flags() & pygame.RLEACCEL)
        image.set_alpha(90)
        self.assertTrue(not image.get_flags() & pygame.RLEACCELOK)
        self.assertTrue(not image.get_flags() & pygame.RLEACCEL)
    finally:
        pygame.display.quit()