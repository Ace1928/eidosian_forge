import os
import os.path
import sys
import unittest
from pygame.tests.test_utils import example_path
import pygame, pygame.image, pygame.pkgdata
def test_save_unknown_extension(self):
    im = pygame.Surface((10, 10), 0, 32)
    s = 'foo.bar'
    self.assertRaises(pygame.error, imageext.save_extended, im, s)