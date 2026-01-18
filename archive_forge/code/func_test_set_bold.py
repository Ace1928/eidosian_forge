from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_set_bold(self):
    f = pygame_font.Font(None, 20)
    self.assertFalse(f.get_bold())
    f.set_bold(True)
    self.assertTrue(f.get_bold())
    f.set_bold(False)
    self.assertFalse(f.get_bold())