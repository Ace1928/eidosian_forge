from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_set_italic(self):
    f = pygame_font.Font(None, 20)
    self.assertFalse(f.get_italic())
    f.set_italic(True)
    self.assertTrue(f.get_italic())
    f.set_italic(False)
    self.assertFalse(f.get_italic())