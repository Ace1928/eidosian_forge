import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_scale2x(self):
    w, h = (32, 32)
    s = pygame.Surface((w, h), pygame.SRCALPHA, 32)
    s1 = pygame.transform.scale2x(s)
    s2 = pygame.transform.scale2x(surface=s)
    self.assertEqual(s1.get_rect().size, (64, 64))
    self.assertEqual(s2.get_rect().size, (64, 64))