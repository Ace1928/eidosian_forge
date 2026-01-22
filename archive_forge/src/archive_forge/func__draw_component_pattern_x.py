from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def _draw_component_pattern_x(self, mask, size, pos, inverse=False):
    pattern = pygame.mask.Mask((size, size))
    ymax = size - 1
    for y in range(size):
        for x in range(size):
            if x in [y, ymax - y]:
                pattern.set_at((x, y))
    if inverse:
        mask.erase(pattern, pos)
        pattern.invert()
    else:
        mask.draw(pattern, pos)
    return pattern