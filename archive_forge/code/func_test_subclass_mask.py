from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_mask(self):
    """Ensures the Mask class can be subclassed."""
    mask = SubMask((5, 3), fill=True)
    self.assertIsInstance(mask, pygame.mask.Mask)
    self.assertIsInstance(mask, SubMask)
    self.assertTrue(mask.test_attribute)