import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_threshold__honors_third_surface(self):
    w, h = size = (32, 32)
    threshold = (20, 20, 20, 20)
    original_color = (25, 25, 25, 25)
    threshold_color = (10, 10, 10, 10)
    original_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
    dest_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
    third_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
    original_surface.fill(original_color)
    third_surface.fill(threshold_color)
    pixels_within_threshold = pygame.transform.threshold(dest_surface=None, surface=original_surface, search_color=threshold_color, threshold=threshold, set_color=None, set_behavior=0)
    self.assertEqual(w * h, pixels_within_threshold)
    pixels_within_threshold = pygame.transform.threshold(dest_surface=None, surface=original_surface, search_color=None, threshold=threshold, set_color=None, set_behavior=0, search_surf=third_surface)
    self.assertEqual(w * h, pixels_within_threshold)