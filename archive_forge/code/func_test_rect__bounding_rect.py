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
def test_rect__bounding_rect(self):
    """Ensures draw rect returns the correct bounding rect.

        Tests rects on and off the surface and a range of width/thickness
        values.
        """
    rect_color = pygame.Color('red')
    surf_color = pygame.Color('black')
    min_width = min_height = 5
    max_width = max_height = 7
    sizes = ((min_width, min_height), (max_width, max_height))
    surface = pygame.Surface((20, 20), 0, 32)
    surf_rect = surface.get_rect()
    big_rect = surf_rect.inflate(min_width * 2 + 1, min_height * 2 + 1)
    for pos in rect_corners_mids_and_center(surf_rect) + rect_corners_mids_and_center(big_rect):
        for attr in RECT_POSITION_ATTRIBUTES:
            for width, height in sizes:
                rect = pygame.Rect((0, 0), (width, height))
                setattr(rect, attr, pos)
                for thickness in range(4):
                    surface.fill(surf_color)
                    bounding_rect = self.draw_rect(surface, rect_color, rect, thickness)
                    expected_rect = create_bounding_rect(surface, surf_color, rect.topleft)
                    self.assertEqual(bounding_rect, expected_rect, f'thickness={thickness}')