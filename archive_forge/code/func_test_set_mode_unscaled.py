import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display
def test_set_mode_unscaled(self):
    """Ensures a window created with SCALED can become smaller."""
    screen = pygame.display.set_mode((300, 300), pygame.SCALED)
    self.assertEqual(screen.get_size(), (300, 300))
    screen = pygame.display.set_mode((200, 200))
    self.assertEqual(screen.get_size(), (200, 200))