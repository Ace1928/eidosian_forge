from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__invalid_setcolor_formats(self):
    """Ensures to_surface handles invalid setcolor formats correctly."""
    mask = pygame.mask.Mask((5, 3))
    for setcolor in ('green color', '#00FF00FF0', '0x00FF00FF0', (1, 2)):
        with self.assertRaises(ValueError):
            mask.to_surface(setcolor=setcolor)
    for setcolor in (pygame.Surface((1, 2)), pygame.Mask((2, 1)), 1.1):
        with self.assertRaises(TypeError):
            mask.to_surface(setcolor=setcolor)