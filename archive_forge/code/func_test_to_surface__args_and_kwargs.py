from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__args_and_kwargs(self):
    """Ensures to_surface accepts a combination of args/kwargs"""
    size = (5, 3)
    surface_color = pygame.Color('red')
    setsurface_color = pygame.Color('yellow')
    unsetsurface_color = pygame.Color('blue')
    setcolor = pygame.Color('green')
    unsetcolor = pygame.Color('cyan')
    surface = pygame.Surface(size, SRCALPHA, 32)
    setsurface = surface.copy()
    unsetsurface = surface.copy()
    setsurface.fill(setsurface_color)
    unsetsurface.fill(unsetsurface_color)
    mask = pygame.mask.Mask(size, fill=True)
    expected_color = setsurface_color
    test_data = ((None, None), ('surface', surface), ('setsurface', setsurface), ('unsetsurface', unsetsurface), ('setcolor', setcolor), ('unsetcolor', unsetcolor), ('dest', (0, 0)))
    args = []
    kwargs = dict(test_data)
    for name, value in test_data:
        if name is not None:
            args.append(value)
        kwargs.pop(name)
        surface.fill(surface_color)
        to_surface = mask.to_surface(*args, **kwargs)
        assertSurfaceFilled(self, to_surface, expected_color)