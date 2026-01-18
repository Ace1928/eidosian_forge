import collections
import time
import unittest
import os
import pygame
@unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'requires the SDL_VIDEODRIVER to be a non dummy value')
@unittest.skipIf(pygame.get_sdl_version() < (2, 0, 16), 'Needs at least SDL 2.0.16')
def test_set_keyboard_grab_and_get_keyboard_grab(self):
    """Ensure set_keyboard_grab() and get_keyboard_grab() work as expected"""
    surf = pygame.display.set_mode((10, 10))
    pygame.event.set_keyboard_grab(True)
    self.assertTrue(pygame.event.get_keyboard_grab())
    pygame.event.set_keyboard_grab(False)
    self.assertFalse(pygame.event.get_keyboard_grab())