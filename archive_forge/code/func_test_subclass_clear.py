from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_clear(self):
    """Ensures clear works for subclassed Masks."""
    mask_size = (4, 3)
    expected_count = 0
    mask = SubMask(mask_size, True)
    mask.clear()
    self.assertEqual(mask.count(), expected_count)