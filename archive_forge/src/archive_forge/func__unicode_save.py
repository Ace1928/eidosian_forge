import os
import os.path
import sys
import unittest
from pygame.tests.test_utils import example_path
import pygame, pygame.image, pygame.pkgdata
def _unicode_save(self, temp_file):
    im = pygame.Surface((10, 10), 0, 32)
    try:
        with open(temp_file, 'w') as f:
            pass
        os.remove(temp_file)
    except OSError:
        raise unittest.SkipTest('the path cannot be opened')
    self.assertFalse(os.path.exists(temp_file))
    try:
        imageext.save_extended(im, temp_file)
        self.assertGreater(os.path.getsize(temp_file), 10)
    finally:
        try:
            os.remove(temp_file)
        except OSError:
            pass