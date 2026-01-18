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
def test_rect__valid_color_formats(self):
    """Ensures draw rect accepts different color formats."""
    pos = (1, 1)
    red_color = pygame.Color('red')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((3, 4))
    kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (1, 1)), 'width': 3}
    reds = ((255, 0, 0), (255, 0, 0, 255), surface.map_rgb(red_color), red_color)
    for color in reds:
        surface.fill(surface_color)
        kwargs['color'] = color
        if isinstance(color, int):
            expected_color = surface.unmap_rgb(color)
        else:
            expected_color = red_color
        bounds_rect = self.draw_rect(**kwargs)
        self.assertEqual(surface.get_at(pos), expected_color)
        self.assertIsInstance(bounds_rect, pygame.Rect)