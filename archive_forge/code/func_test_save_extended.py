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
def test_save_extended(self):
    surf = pygame.Surface((5, 5))
    surf.fill((23, 23, 23))
    passing_formats = ['jpg', 'png']
    passing_formats += [fmt.upper() for fmt in passing_formats]
    magic_hex = {}
    magic_hex['jpg'] = [255, 216, 255, 224]
    magic_hex['png'] = [137, 80, 78, 71]
    failing_formats = ['bmp', 'tga']
    failing_formats += [fmt.upper() for fmt in failing_formats]
    for fmt in passing_formats:
        temp_file_name = f'temp_file.{fmt}'
        pygame.image.save_extended(surf, temp_file_name)
        with open(temp_file_name, 'rb') as file:
            self.assertEqual(1, test_magic(file, magic_hex[fmt.lower()]))
        loaded_file = pygame.image.load(temp_file_name)
        self.assertEqual(loaded_file.get_at((0, 0)), surf.get_at((0, 0)))
        os.remove(temp_file_name)
    for fmt in failing_formats:
        self.assertRaises(pygame.error, pygame.image.save_extended, surf, f'temp_file.{fmt}')