import unittest
import os
import platform
import warnings
import pygame
def test_set_visible(self):
    """Ensures set_visible returns the correct values."""
    pygame.mouse.set_visible(True)
    for expected_visible in (False, True):
        prev_visible = pygame.mouse.set_visible(expected_visible)
        self.assertEqual(prev_visible, not expected_visible)