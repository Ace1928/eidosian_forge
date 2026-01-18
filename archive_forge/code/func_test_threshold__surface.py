import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_threshold__surface(self):
    """ """
    from pygame.transform import threshold
    s1 = pygame.Surface((32, 32), SRCALPHA, 32)
    s2 = pygame.Surface((32, 32), SRCALPHA, 32)
    s3 = pygame.Surface((1, 1), SRCALPHA, 32)
    THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF = 2
    s1.fill((40, 40, 40))
    s1.set_at((0, 0), (170, 170, 170))
    THRESHOLD_BEHAVIOR_COUNT = 0
    num_threshold_pixels = threshold(dest_surface=None, surface=s1, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=None, set_behavior=THRESHOLD_BEHAVIOR_COUNT)
    self.assertEqual(num_threshold_pixels, s1.get_height() * s1.get_width() - 1)
    s1.fill((254, 254, 254))
    s2.fill((255, 255, 255))
    s3.fill((255, 255, 255))
    s1.set_at((0, 0), (170, 170, 170))
    num_threshold_pixels = threshold(None, s1, (254, 254, 254), (1, 1, 1), None, THRESHOLD_BEHAVIOR_COUNT)
    self.assertEqual(num_threshold_pixels, s1.get_height() * s1.get_width() - 1)
    num_threshold_pixels = threshold(None, s1, None, (1, 1, 1), None, THRESHOLD_BEHAVIOR_COUNT, s2)
    self.assertEqual(num_threshold_pixels, s1.get_height() * s1.get_width() - 1)
    num_threshold_pixels = threshold(None, s1, (253, 253, 253), (0, 0, 0), None, THRESHOLD_BEHAVIOR_COUNT)
    self.assertEqual(num_threshold_pixels, 0)
    num_threshold_pixels = threshold(None, s1, None, (0, 0, 0), None, THRESHOLD_BEHAVIOR_COUNT, s2)
    self.assertEqual(num_threshold_pixels, 0)