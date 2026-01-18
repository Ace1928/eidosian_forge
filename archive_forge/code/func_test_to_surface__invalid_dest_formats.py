from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__invalid_dest_formats(self):
    """Ensures to_surface handles invalid dest formats correctly."""
    mask = pygame.mask.Mask((3, 5))
    invalid_dests = ((0,), (0, 0, 0), {0, 1}, {0: 1}, Rect)
    for dest in invalid_dests:
        with self.assertRaises(TypeError):
            mask.to_surface(dest=dest)