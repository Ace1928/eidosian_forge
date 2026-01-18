from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_set_at(self):
    """Ensures set_at works for subclassed Masks."""
    expected_bit = 1
    expected_count = 1
    pos = (0, 0)
    mask = SubMask(fill=False, size=(4, 2))
    mask.set_at(pos)
    self.assertEqual(mask.get_at(pos), expected_bit)
    self.assertEqual(mask.count(), expected_count)