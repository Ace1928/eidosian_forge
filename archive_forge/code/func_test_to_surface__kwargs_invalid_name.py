from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__kwargs_invalid_name(self):
    """Ensures to_surface detects invalid kwarg names."""
    mask = pygame.mask.Mask((3, 2))
    kwargs = {'setcolour': pygame.Color('red')}
    with self.assertRaises(TypeError):
        mask.to_surface(**kwargs)