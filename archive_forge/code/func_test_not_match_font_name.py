from re import T
import sys
import os
import unittest
import pathlib
import platform
import pygame
from pygame import font as pygame_font  # So font can be replaced with ftfont
def test_not_match_font_name(self):
    """match_font return None when names of various types do not exist"""
    not_a_font = 'thisisnotafont'
    not_a_font_b = b'thisisnotafont'
    bad_font_names = [not_a_font, ','.join([not_a_font, not_a_font, not_a_font]), [not_a_font, not_a_font, not_a_font], (name for name in [not_a_font, not_a_font, not_a_font]), not_a_font_b, b','.join([not_a_font_b, not_a_font_b, not_a_font_b]), [not_a_font_b, not_a_font_b, not_a_font_b], [not_a_font, not_a_font_b, not_a_font]]
    for font_name in bad_font_names:
        self.assertIsNone(pygame_font.match_font(font_name), font_name)