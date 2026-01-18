import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_rotate_of_0_sized_surface(self):
    canvas1 = pygame.Surface((0, 1))
    canvas2 = pygame.Surface((1, 0))
    pygame.transform.rotate(canvas1, 42)
    pygame.transform.rotate(canvas2, 42)