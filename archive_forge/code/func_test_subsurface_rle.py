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
def test_subsurface_rle(self):
    """Ensure an RLE sub-surface works independently of its parent."""
    color = (250, 25, 25, 255)
    color2 = (200, 200, 250, 255)
    sub_rect = pygame.Rect(16, 16, 16, 16)
    s0 = pygame.Surface((32, 32), 24)
    s1 = pygame.Surface((32, 32), 24)
    s1.set_colorkey((255, 0, 255), pygame.RLEACCEL)
    s1.fill(color)
    s2 = s1.subsurface(sub_rect)
    s2.fill(color2)
    s0.blit(s1, (0, 0))
    self.assertTrue(s1.get_flags() & pygame.RLEACCEL)
    self.assertTrue(not s2.get_flags() & pygame.RLEACCEL)