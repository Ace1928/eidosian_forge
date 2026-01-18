import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_rotozoom(self):
    s = pygame.Surface((10, 0))
    pygame.transform.scale(s, (10, 2))
    s1 = pygame.transform.rotozoom(s, 30, 1)
    s2 = pygame.transform.rotozoom(surface=s, angle=30, scale=1)
    self.assertEqual(s1.get_rect(), pygame.Rect(0, 0, 0, 0))
    self.assertEqual(s2.get_rect(), pygame.Rect(0, 0, 0, 0))