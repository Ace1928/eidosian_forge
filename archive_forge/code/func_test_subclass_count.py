from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_count(self):
    """Ensures count works for subclassed Masks."""
    mask_size = (5, 2)
    expected_count = mask_size[0] * mask_size[1] - 1
    mask = SubMask(fill=True, size=mask_size)
    mask.set_at((1, 1), 0)
    count = mask.count()
    self.assertEqual(count, expected_count)