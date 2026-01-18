import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_threshold__uneven_colors(self):
    w, h = size = (16, 16)
    original_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
    dest_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
    original_surface.fill(0)
    threshold_color_template = [5, 5, 5, 5]
    threshold_template = [6, 6, 6, 6]
    for pos in range(len('rgb')):
        threshold_color = threshold_color_template[:]
        threshold = threshold_template[:]
        threshold_color[pos] = 45
        threshold[pos] = 50
        pixels_within_threshold = pygame.transform.threshold(None, original_surface, threshold_color, threshold, set_color=None, set_behavior=0)
        self.assertEqual(w * h, pixels_within_threshold)