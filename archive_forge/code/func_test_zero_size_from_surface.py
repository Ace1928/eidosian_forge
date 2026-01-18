from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_size_from_surface(self):
    """Ensures from_surface can create masks from zero sized surfaces."""
    for size in ((100, 0), (0, 100), (0, 0)):
        mask = pygame.mask.from_surface(pygame.Surface(size))
        self.assertIsInstance(mask, pygame.mask.MaskType, f'size={size}')
        self.assertEqual(mask.get_size(), size)