from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_connected_component(self):
    """Ensures connected_component works for subclassed Masks."""
    expected_count = 0
    expected_size = (3, 4)
    mask = SubMask(expected_size)
    cc_mask = mask.connected_component()
    self.assertIsInstance(cc_mask, pygame.mask.Mask)
    self.assertNotIsInstance(cc_mask, SubMask)
    self.assertEqual(cc_mask.count(), expected_count)
    self.assertEqual(cc_mask.get_size(), expected_size)