from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_outline(self):
    """Ensures outline works for subclassed Masks."""
    expected_outline = []
    mask = SubMask((3, 4))
    outline = mask.outline()
    self.assertListEqual(outline, expected_outline)