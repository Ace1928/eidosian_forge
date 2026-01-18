import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def test_set_mode(self):
    """Ensures set_mode works as expected."""
    scrap.set_mode(pygame.SCRAP_SELECTION)
    scrap.set_mode(pygame.SCRAP_CLIPBOARD)
    self.assertRaises(ValueError, scrap.set_mode, 1099)