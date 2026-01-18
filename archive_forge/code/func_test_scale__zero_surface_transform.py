import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_scale__zero_surface_transform(self):
    tmp_surface = pygame.transform.scale(pygame.Surface((128, 128)), (0, 0))
    self.assertEqual(tmp_surface.get_size(), (0, 0))
    tmp_surface = pygame.transform.scale(tmp_surface, (128, 128))
    self.assertEqual(tmp_surface.get_size(), (128, 128))