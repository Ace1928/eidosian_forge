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
def test_set_at(self):
    s = pygame.Surface((100, 100), 0, 24)
    s.fill((0, 0, 0))
    s.set_at((0, 0), (10, 10, 10, 255))
    r = s.get_at((0, 0))
    self.assertIsInstance(r, pygame.Color)
    self.assertEqual(r, (10, 10, 10, 255))
    s.fill((0, 0, 0, 255))
    s.set_at((10, 1), 255)
    r = s.get_at((10, 1))
    self.assertEqual(r, (0, 0, 255, 255))