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
def test_set_clip(self):
    """see if surface.set_clip(None) works correctly."""
    s = pygame.Surface((800, 600))
    r = pygame.Rect(10, 10, 10, 10)
    s.set_clip(r)
    r.move_ip(10, 0)
    s.set_clip(None)
    res = s.get_clip()
    self.assertEqual(res[0], 0)
    self.assertEqual(res[2], 800)