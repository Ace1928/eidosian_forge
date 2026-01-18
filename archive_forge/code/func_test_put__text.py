import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
def test_put__text(self):
    """Ensures put can place text into the clipboard."""
    scrap.put(pygame.SCRAP_TEXT, b'Hello world')
    self.assertEqual(scrap.get(pygame.SCRAP_TEXT), b'Hello world')
    scrap.put(pygame.SCRAP_TEXT, b'Another String')
    self.assertEqual(scrap.get(pygame.SCRAP_TEXT), b'Another String')