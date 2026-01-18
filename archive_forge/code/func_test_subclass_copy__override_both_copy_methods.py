from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_copy__override_both_copy_methods(self):
    """Ensures copy works for subclassed Masks overriding copy/__copy__."""
    mask = SubMaskCopyAndDunderCopy((65, 2), fill=True)
    for mask_copy in (mask.copy(), copy.copy(mask)):
        self.assertIsInstance(mask_copy, pygame.mask.Mask)
        self.assertIsInstance(mask_copy, SubMaskCopyAndDunderCopy)
        self.assertIsNot(mask_copy, mask)
        assertMaskEqual(self, mask_copy, mask)
        self.assertTrue(mask_copy.test_attribute)