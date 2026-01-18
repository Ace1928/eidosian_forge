import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_threshold__subclassed_surface(self):
    """Ensure threshold accepts subclassed surfaces."""
    expected_size = (13, 11)
    expected_flags = 0
    expected_depth = 32
    expected_color = (90, 80, 70, 255)
    expected_count = 0
    surface = test_utils.SurfaceSubclass(expected_size, expected_flags, expected_depth)
    dest_surface = test_utils.SurfaceSubclass(expected_size, expected_flags, expected_depth)
    search_surface = test_utils.SurfaceSubclass(expected_size, expected_flags, expected_depth)
    surface.fill((10, 10, 10))
    dest_surface.fill((255, 255, 255))
    search_surface.fill((20, 20, 20))
    count = pygame.transform.threshold(dest_surface=dest_surface, surface=surface, threshold=(1, 1, 1), set_color=expected_color, search_color=None, search_surf=search_surface)
    self.assertIsInstance(dest_surface, pygame.Surface)
    self.assertIsInstance(dest_surface, test_utils.SurfaceSubclass)
    self.assertEqual(count, expected_count)
    self.assertEqual(dest_surface.get_at((0, 0)), expected_color)
    self.assertEqual(dest_surface.get_bitsize(), expected_depth)
    self.assertEqual(dest_surface.get_size(), expected_size)
    self.assertEqual(dest_surface.get_flags(), expected_flags)