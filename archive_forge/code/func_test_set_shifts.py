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
def test_set_shifts(self):
    s = pygame.Surface((32, 32))
    r, g, b, a = s.get_shifts()
    self.assertRaises(TypeError, s.set_shifts, (b, g, r, a))