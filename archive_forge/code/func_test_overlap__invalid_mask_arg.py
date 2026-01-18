from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
@unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
def test_overlap__invalid_mask_arg(self):
    """Ensure overlap handles invalid mask arguments correctly."""
    size = (5, 3)
    offset = (0, 0)
    mask = pygame.mask.Mask(size)
    invalid_mask = pygame.Surface(size)
    with self.assertRaises(TypeError):
        overlap_pos = mask.overlap(invalid_mask, offset)