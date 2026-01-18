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
def test_solarwolf_rle_usage(self):
    """Test for error/crash when calling set_colorkey() followed
        by convert twice in succession. Code originally taken
        from solarwolf."""

    def optimize(img):
        clear = img.get_colorkey()
        img.set_colorkey(clear, RLEACCEL)
        self.assertEqual(img.get_colorkey(), clear)
        return img.convert()
    pygame.display.init()
    try:
        pygame.display.set_mode((640, 480))
        image = pygame.image.load(example_path(os.path.join('data', 'alien1.png')))
        image = image.convert()
        orig_colorkey = image.get_colorkey()
        image = optimize(image)
        image = optimize(image)
        self.assertTrue(image.get_flags() & pygame.RLEACCELOK)
        self.assertTrue(not image.get_flags() & pygame.RLEACCEL)
        self.assertEqual(image.get_colorkey(), orig_colorkey)
        self.assertTrue(isinstance(image, pygame.Surface))
    finally:
        pygame.display.quit()