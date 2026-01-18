from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_scale(self):
    """Ensures scale works for subclassed Masks."""
    expected_size = (5, 2)
    mask = SubMask((1, 4))
    scaled_mask = mask.scale(expected_size)
    self.assertIsInstance(scaled_mask, pygame.mask.Mask)
    self.assertNotIsInstance(scaled_mask, SubMask)
    self.assertEqual(scaled_mask.get_size(), expected_size)