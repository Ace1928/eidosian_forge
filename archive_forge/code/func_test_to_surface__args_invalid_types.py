from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__args_invalid_types(self):
    """Ensures to_surface detects invalid kwarg types."""
    size = (3, 2)
    mask = pygame.mask.Mask(size, fill=True)
    invalid_surf = pygame.Color('green')
    invalid_color = pygame.Surface(size)
    with self.assertRaises(TypeError):
        mask.to_surface(None, None, None, None, None, (0,))
    with self.assertRaises(TypeError):
        mask.to_surface(None, None, None, None, invalid_color)
    with self.assertRaises(TypeError):
        mask.to_surface(None, None, None, invalid_color, None)
    with self.assertRaises(TypeError):
        mask.to_surface(None, None, invalid_surf, None, None)
    with self.assertRaises(TypeError):
        mask.to_surface(None, invalid_surf, None, None, None)
    with self.assertRaises(TypeError):
        mask.to_surface(invalid_surf, None, None, None, None)