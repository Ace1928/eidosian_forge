import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_smoothscale(self):
    """Tests the stated boundaries, sizing, and color blending of smoothscale function"""

    def smoothscale_low_bpp():
        starting_surface = pygame.Surface((20, 20), depth=12)
        smoothscaled_surface = pygame.transform.smoothscale(starting_surface, (10, 10))
    self.assertRaises(ValueError, smoothscale_low_bpp)

    def smoothscale_high_bpp():
        starting_surface = pygame.Surface((20, 20), depth=48)
        smoothscaled_surface = pygame.transform.smoothscale(starting_surface, (10, 10))
    self.assertRaises(ValueError, smoothscale_high_bpp)

    def smoothscale_invalid_scale():
        starting_surface = pygame.Surface((20, 20), depth=32)
        smoothscaled_surface = pygame.transform.smoothscale(starting_surface, (-1, -1))
    self.assertRaises(ValueError, smoothscale_invalid_scale)
    two_pixel_surface = pygame.Surface((2, 1), depth=32)
    two_pixel_surface.fill(pygame.Color(0, 0, 0), pygame.Rect(0, 0, 1, 1))
    two_pixel_surface.fill(pygame.Color(255, 255, 255), pygame.Rect(1, 0, 1, 1))
    for k in [2 ** x for x in range(5, 8)]:
        bigger_surface = pygame.transform.smoothscale(two_pixel_surface, (k, 1))
        self.assertEqual(bigger_surface.get_at((k // 2, 0)), pygame.Color(127, 127, 127))
        self.assertEqual(bigger_surface.get_size(), (k, 1))
    two_five_six_surf = pygame.Surface((256, 1), depth=32)
    two_five_six_surf.fill(pygame.Color(0, 0, 0), pygame.Rect(0, 0, 128, 1))
    two_five_six_surf.fill(pygame.Color(255, 255, 255), pygame.Rect(128, 0, 128, 1))
    for k in range(3, 11, 2):
        smaller_surface = pygame.transform.smoothscale(two_five_six_surf, (k, 1))
        self.assertEqual(smaller_surface.get_at((k // 2, 0)), pygame.Color(127, 127, 127))
        self.assertEqual(smaller_surface.get_size(), (k, 1))