import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_scale__alpha(self):
    """see if set_alpha information is kept."""
    s = pygame.Surface((32, 32))
    s.set_alpha(55)
    self.assertEqual(s.get_alpha(), 55)
    s = pygame.Surface((32, 32))
    s.set_alpha(55)
    s2 = pygame.transform.scale(s, (64, 64))
    s3 = s.copy()
    self.assertEqual(s.get_alpha(), s3.get_alpha())
    self.assertEqual(s.get_alpha(), s2.get_alpha())