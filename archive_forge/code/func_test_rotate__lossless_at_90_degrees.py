import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_rotate__lossless_at_90_degrees(self):
    w, h = (32, 32)
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    gradient = list(test_utils.gradient(w, h))
    for pt, color in gradient:
        s.set_at(pt, color)
    for rotation in (90, -90):
        s = pygame.transform.rotate(s, rotation)
    for pt, color in gradient:
        self.assertTrue(s.get_at(pt) == color)