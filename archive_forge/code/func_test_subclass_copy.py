from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_copy(self):
    """Ensures copy works for subclassed Masks."""
    mask = SubMask((65, 2), fill=True)
    for mask_copy in (mask.copy(), copy.copy(mask)):
        self.assertIsInstance(mask_copy, pygame.mask.Mask)
        self.assertIsInstance(mask_copy, SubMask)
        self.assertIsNot(mask_copy, mask)
        assertMaskEqual(self, mask_copy, mask)
        self.assertFalse(hasattr(mask_copy, 'test_attribute'))