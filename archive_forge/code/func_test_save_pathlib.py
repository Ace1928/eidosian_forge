import array
import binascii
import io
import os
import tempfile
import unittest
import glob
import pathlib
from pygame.tests.test_utils import example_path, png, tostring
import pygame, pygame.image, pygame.pkgdata
def test_save_pathlib(self):
    surf = pygame.Surface((1, 1))
    surf.fill((23, 23, 23))
    with tempfile.NamedTemporaryFile(suffix='.tga', delete=False) as f:
        temp_filename = f.name
    path = pathlib.Path(temp_filename)
    try:
        pygame.image.save(surf, path)
        s2 = pygame.image.load(path)
        self.assertEqual(s2.get_at((0, 0)), surf.get_at((0, 0)))
    finally:
        os.remove(temp_filename)