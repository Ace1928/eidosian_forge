from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_connected_component__indexed(self):
    """Ensures connected_component correctly handles zero sized masks
        when using an index argument."""
    for size in ((91, 0), (0, 90), (0, 0)):
        mask = pygame.mask.Mask(size)
        with self.assertRaises(IndexError):
            cc_mask = mask.connected_component((0, 0))