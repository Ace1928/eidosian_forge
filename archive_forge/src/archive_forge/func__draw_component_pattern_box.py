from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def _draw_component_pattern_box(self, mask, size, pos, inverse=False):
    pattern = pygame.mask.Mask((size, size), fill=True)
    pattern.set_at((size // 2, size // 2), 0)
    if inverse:
        mask.erase(pattern, pos)
        pattern.invert()
    else:
        mask.draw(pattern, pos)
    return pattern