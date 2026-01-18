from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_overlap__invalid_offset_arg(self):
    """Ensure overlap handles invalid offset arguments correctly."""
    size = (2, 7)
    offset = '(0, 0)'
    mask1 = pygame.mask.Mask(size)
    mask2 = pygame.mask.Mask(size)
    with self.assertRaises(TypeError):
        overlap_pos = mask1.overlap(mask2, offset)