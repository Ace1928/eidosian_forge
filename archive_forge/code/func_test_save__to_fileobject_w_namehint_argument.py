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
def test_save__to_fileobject_w_namehint_argument(self):
    s = pygame.Surface((10, 10))
    s.fill((23, 23, 23))
    magic_hex = {}
    magic_hex['jpg'] = [255, 216, 255, 224]
    magic_hex['png'] = [137, 80, 78, 71]
    magic_hex['bmp'] = [66, 77]
    formats = ['tga', 'jpg', 'bmp', 'png']
    formats = formats + [x.upper() for x in formats]
    SDL_Im_version = pygame.image.get_sdl_image_version()
    isAtLeastSDL_image_2_0_2 = SDL_Im_version is not None and SDL_Im_version[0] * 10000 + SDL_Im_version[1] * 100 + SDL_Im_version[2] >= 20002
    for fmt in formats:
        tmp_file, tmp_filename = tempfile.mkstemp(suffix=f'.{fmt}')
        if not isAtLeastSDL_image_2_0_2 and fmt.lower() == 'jpg':
            with os.fdopen(tmp_file, 'wb') as handle:
                with self.assertRaises(pygame.error):
                    pygame.image.save(s, handle, tmp_filename)
        else:
            with os.fdopen(tmp_file, 'r+b') as handle:
                pygame.image.save(s, handle, tmp_filename)
                if fmt.lower() in magic_hex:
                    handle.seek(0)
                    self.assertEqual((1, fmt), (test_magic(handle, magic_hex[fmt.lower()]), fmt))
                handle.flush()
                handle.seek(0)
                s2 = pygame.image.load(handle, tmp_filename)
                self.assertEqual(s2.get_at((0, 0)), s.get_at((0, 0)))
        os.remove(tmp_filename)