import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
@staticmethod
def _create_surfaces():
    surfaces = []
    for size in ((49, 49), (50, 50)):
        for depth in (8, 16, 24, 32):
            for flags in (0, SRCALPHA):
                surface = pygame.display.set_mode(size, flags, depth)
                surfaces.append(surface)
                surfaces.append(surface.convert_alpha())
    return surfaces