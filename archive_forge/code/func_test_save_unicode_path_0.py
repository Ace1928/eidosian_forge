import os
import os.path
import sys
import unittest
from pygame.tests.test_utils import example_path
import pygame, pygame.image, pygame.pkgdata
def test_save_unicode_path_0(self):
    """unicode object with ASCII chars"""
    self._unicode_save('temp_file.png')